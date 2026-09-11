"""Durable, serialized execution of claimed worker operations."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

from pydantic import ValidationError

from .compute import ComputeError, IndexRepository
from .documents import DocumentStore, DocumentStoreError
from .jobs import JobStore, StoredOperation
from .model_controller import ModelController, ModelControllerError, ModelReadinessProbe
from .protocol import (
    IndexJobResult,
    IngestJobResult,
    JobProgressDetail,
    JobOperation,
    ModelJobResult,
    ModelRuntimeState,
    PageDocument,
    QueryProgressStage,
    QueryStrategy,
)


class WorkerExecutor:
    """One runner per worker; SQLite claiming also prevents cross-process overlap."""

    def __init__(
        self,
        *,
        environment: str,
        job_store: JobStore,
        document_store: DocumentStore,
        index_repository: IndexRepository,
        model_controller: ModelController,
        model_readiness_probe: ModelReadinessProbe,
        approved_models: tuple[str, ...] = (),
        model_start_timeout_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
        poll_seconds: float = 0.5,
        idle_maintenance: Callable[[], None] | None = None,
        maintenance_interval_seconds: float = 60.0,
    ) -> None:
        self.environment = environment
        self.job_store = job_store
        self.document_store = document_store
        self.index_repository = index_repository
        self.model_controller = model_controller
        self.model_readiness_probe = model_readiness_probe
        self.approved_models = tuple(approved_models)
        self.model_start_timeout_seconds = max(1.0, float(model_start_timeout_seconds))
        self.clock = clock
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.idle_maintenance = idle_maintenance
        self.maintenance_interval_seconds = max(1.0, float(maintenance_interval_seconds))
        self._last_maintenance_attempt = float("-inf")
        self._maintenance_healthy = True
        self._condition = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def maintenance_healthy(self) -> bool:
        return self._maintenance_healthy

    def start(self) -> None:
        with self._condition:
            if self.running:
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name=f"novo-chat-worker-{self.environment}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_seconds: float = 10.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_seconds))

    def notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
            if self.run_once():
                continue
            self.run_idle_maintenance()
            with self._condition:
                if self._stopping:
                    return
                self._condition.wait(timeout=self.poll_seconds)

    def run_idle_maintenance(self, *, force: bool = False) -> bool:
        if self.idle_maintenance is None:
            return True
        now = float(self.clock())
        if not force and now - self._last_maintenance_attempt < self.maintenance_interval_seconds:
            return self._maintenance_healthy
        self._last_maintenance_attempt = now
        try:
            self.idle_maintenance()
        except Exception:
            # Health reports the failed retention pass while the runner stays
            # alive to retry. Request/job details are deliberately not logged.
            self._maintenance_healthy = False
        else:
            self._maintenance_healthy = True
        return self._maintenance_healthy

    def run_until_idle(self, *, max_operations: int = 10_000) -> int:
        completed = 0
        while completed < max_operations and self.run_once():
            completed += 1
        return completed

    def run_once(self) -> bool:
        operation = self.job_store.claim_next_operation(environment=self.environment, now=self.clock())
        if operation is None:
            return False
        try:
            result = self._execute(operation)
            self.job_store.succeed(operation.operation_id, result, now=self.clock())
        except (DocumentStoreError, ComputeError, ModelControllerError) as exc:
            self.job_store.fail(
                operation.operation_id,
                error_code=exc.code,
                error_message=exc.message,
                retryable=exc.retryable if hasattr(exc, "retryable") else False,
                now=self.clock(),
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            self.job_store.fail(
                operation.operation_id,
                error_code="INVALID_OPERATION_PAYLOAD",
                error_message="Stored operation payload is invalid.",
                retryable=False,
                now=self.clock(),
            )
        except Exception:
            self.job_store.fail(
                operation.operation_id,
                error_code="OPERATION_FAILED",
                error_message="Worker operation failed.",
                retryable=True,
                now=self.clock(),
            )
        return True

    def _execute(self, operation: StoredOperation) -> Mapping[str, Any]:
        if operation.operation is JobOperation.INGEST_BATCH:
            return self._ingest_batch(operation)
        if operation.operation is JobOperation.INGEST_FINALIZE:
            return self._ingest_finalize(operation)
        if operation.operation is JobOperation.INDEX_REBUILD:
            return self._rebuild(operation)
        if operation.operation is JobOperation.QUERY:
            return self._query(operation)
        if operation.operation is JobOperation.MODEL_START:
            return self._model_lifecycle(operation, start=True)
        if operation.operation is JobOperation.MODEL_STOP:
            return self._model_lifecycle(operation, start=False)
        raise ValueError("unsupported operation")

    @staticmethod
    def _one_scope(operation: StoredOperation):
        if len(operation.scope) != 1:
            raise ValueError("operation requires exactly one notebook scope")
        return operation.scope[0]

    def _ingest_batch(self, operation: StoredOperation) -> Mapping[str, Any]:
        scope = self._one_scope(operation)
        pages = tuple(PageDocument.model_validate(page) for page in operation.payload["pages"])
        accepted = self.document_store.stage_batch(
            scope=scope,
            batch_number=int(operation.payload["batchNumber"]),
            batch_checksum=str(operation.payload["batchChecksum"]),
            pages=pages,
        )
        return IngestJobResult(scope=scope, accepted_pages=accepted).model_dump(mode="json", by_alias=True)

    def _ingest_finalize(self, operation: StoredOperation) -> Mapping[str, Any]:
        scope = self._one_scope(operation)
        finalized = self.document_store.finalize(
            scope=scope,
            document_checksum=str(operation.payload["documentChecksum"]),
            page_count=int(operation.payload["pageCount"]),
            batch_count=int(operation.payload["batchCount"]),
        )
        return IngestJobResult(
            scope=scope,
            accepted_pages=len(finalized.pages),
        ).model_dump(mode="json", by_alias=True)

    def _rebuild(self, operation: StoredOperation) -> Mapping[str, Any]:
        candidates = []
        try:
            for position, scope in enumerate(operation.scope):
                document = self.document_store.load(scope)
                candidate = self.index_repository.build_candidate(
                    document,
                    operation_id=operation.operation_id,
                )
                candidates.append(candidate)
                self.job_store.set_progress(
                    operation.operation_id,
                    0.8 * (position + 1) / max(1, len(operation.scope)),
                    now=self.clock(),
                )
            committed = [self.index_repository.commit_candidate(candidate) for candidate in candidates]
            self.job_store.activate_indexes(
                [(entry.scope, entry.artifact_id) for entry in committed],
                environment=self.environment,
                now=self.clock(),
            )
        except Exception:
            for candidate in candidates:
                self.index_repository.discard_candidate(candidate)
            raise
        return IndexJobResult(
            indexes=tuple(operation.scope),
            chunk_count=sum(candidate.chunk_count for candidate in candidates),
        ).model_dump(mode="json", by_alias=True)

    def _query(self, operation: StoredOperation) -> Mapping[str, Any]:
        artifacts = []
        for scope in operation.scope:
            active = self.job_store.active_index(scope, environment=self.environment)
            if active is None:
                raise ComputeError(
                    "INDEX_NOT_READY",
                    "An exact requested notebook index is not ready.",
                    retryable=True,
                )
            artifacts.append(self.index_repository.load_active(active))
        strategy = QueryStrategy(str(operation.payload.get("strategy") or QueryStrategy.HYBRID.value))
        model = str(operation.payload["model"])
        if not self.model_readiness_probe.is_ready(model):
            raise ComputeError("MODEL_NOT_READY", "Requested model is not ready.", retryable=True)
        self.job_store.set_progress(
            operation.operation_id,
            0.05,
            progress_detail=JobProgressDetail(
                stage=QueryProgressStage.PLANNING
            ).model_dump(mode="json", by_alias=True),
            now=self.clock(),
        )

        def report_progress(detail: JobProgressDetail) -> None:
            progress = {
                QueryProgressStage.SEARCHING: 0.30,
                QueryProgressStage.ANSWERING: 0.65,
            }.get(detail.stage, 0.05)
            self.job_store.set_progress(
                operation.operation_id,
                progress,
                progress_detail=detail.model_dump(mode="json", by_alias=True),
                now=self.clock(),
            )

        result = self.index_repository.query(
            artifacts,
            question=str(operation.payload["question"]),
            retrieval_question=str(operation.payload.get("retrievalQuestion") or operation.payload["question"]),
            model=model,
            strategy=strategy,
            max_sources=int(operation.payload.get("maxSources", 16)),
            retrieval_top_k=int(operation.payload.get("retrievalTopK", 16)),
            progress_callback=report_progress,
        )
        return result.model_dump(mode="json", by_alias=True)

    def _model_lifecycle(self, operation: StoredOperation, *, start: bool) -> Mapping[str, Any]:
        model = str(operation.payload["model"])
        if start:
            for other_model in self.approved_models:
                if other_model == model:
                    continue
                other_status = self.model_controller.status(
                    other_model,
                    request_id=operation.request_id,
                    actor_user_id=operation.actor_user_id,
                )
                if str(other_status.get("state") or "").lower() not in {"running", "ready"}:
                    continue
                stopped = self.model_controller.stop(
                    other_model,
                    request_id=operation.request_id,
                    actor_user_id=operation.actor_user_id,
                )
                if str(stopped.get("state") or "").lower() != "stopped":
                    raise ModelControllerError(
                        "MODEL_SWITCH_FAILED",
                        "The active model could not be stopped safely.",
                    )
        method = self.model_controller.start if start else self.model_controller.stop
        response = method(
            model,
            request_id=operation.request_id,
            actor_user_id=operation.actor_user_id,
        )
        controller_state = str(response.get("state") or "").lower()
        if start:
            if controller_state not in {"running", "ready"}:
                raise ModelControllerError(
                    "MODEL_TRANSITION_FAILED",
                    "Model did not enter a running state.",
                )

            last_progress = 0.0

            def report_start_progress() -> None:
                nonlocal last_progress
                try:
                    status = self.model_controller.status(
                        model,
                        request_id=operation.request_id,
                        actor_user_id=operation.actor_user_id,
                    )
                except ModelControllerError:
                    return
                raw_progress = status.get("progress")
                if (
                    isinstance(raw_progress, bool)
                    or not isinstance(raw_progress, (int, float))
                ):
                    return
                progress = float(raw_progress)
                if not 0.0 <= progress <= 1.0 or progress < last_progress:
                    return
                last_progress = progress
                self.job_store.set_progress(
                    operation.operation_id,
                    progress,
                    now=self.clock(),
                )

            if not self.model_readiness_probe.wait_until_ready(
                model,
                timeout_seconds=self.model_start_timeout_seconds,
                progress_callback=report_start_progress,
            ):
                raise ModelControllerError(
                    "MODEL_READINESS_TIMEOUT",
                    "Model started but did not become ready before the timeout.",
                )
            state = ModelRuntimeState.READY
        else:
            state = ModelRuntimeState.STOPPED if controller_state == "stopped" else ModelRuntimeState.FAILED
        expected = ModelRuntimeState.READY if start else ModelRuntimeState.STOPPED
        if state is not expected:
            raise ModelControllerError(
                "MODEL_TRANSITION_FAILED",
                "Model did not reach the requested state.",
            )
        return ModelJobResult(model=model, state=state).model_dump(mode="json", by_alias=True)


__all__ = ["WorkerExecutor"]
