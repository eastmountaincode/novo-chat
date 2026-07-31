"""Loopback-only, HMAC-authenticated Novo Chat compute worker API."""

from __future__ import annotations

import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.requests import ClientDisconnect

from .compute import ComputeBackend, IndexRepository
from .documents import DocumentStore
from .executor import WorkerExecutor
from .jobs import IdempotencyConflict, JobNotFound, JobStore, QueueFull, StoredJob
from .model_controller import (
    ModelController,
    ModelReadinessProbe,
    UnavailableModelController,
    UnavailableModelReadinessProbe,
)
from .protocol import (
    DEFAULT_AUDIENCE,
    PROTOCOL_VERSION,
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    IndexStatusItem,
    IndexStatusRequest,
    IndexStatusResponse,
    IngestBatchRequest,
    IngestFinalizeRequest,
    JobError,
    JobOperation,
    JobState,
    JobStatusResponse,
    JobSubmissionResponse,
    JobView,
    ModelDisplayDetails,
    ModelJobRequest,
    ModelRuntimeState,
    ModelStatusItem,
    ModelStatusResponse,
    NotebookScope,
    ProtocolError,
    QueryRequest,
    RebuildRequest,
    WorkerState,
    ingest_pages_checksum,
    new_request_id,
    sign_response,
    validate_request_id,
    verify_request,
)


INTERNAL_PREFIX = "/internal/v1"
_JOB_ID = re.compile(r"^job_[A-Za-z0-9_-]{20,80}$")


@dataclass(frozen=True)
class WorkerConfig:
    environment: str
    state_database_path: str | Path
    request_secrets_by_key_id: Mapping[str, bytes | str]
    response_key_id: str
    response_secret: bytes | str
    audience: str = DEFAULT_AUDIENCE
    approved_models: tuple[str, ...] = ()
    model_details: Mapping[str, ModelDisplayDetails] = field(default_factory=dict)
    index_schema_versions: tuple[str, ...] = ("novo-chat-index-v1",)
    max_clock_skew_seconds: int = 60
    max_request_bytes: int = 32 * 1024 * 1024
    max_scope_entries: int = 256
    max_ingest_pages_per_batch: int = 250
    max_active_operations: int = 64
    state_directory: str | Path | None = None
    executor_poll_seconds: float = 0.5
    model_start_timeout_seconds: float = 600.0
    job_retention_seconds: int = 7 * 24 * 60 * 60
    orphan_grace_seconds: int = 60 * 60
    document_storage_quota_bytes: int = 50 * 1024 * 1024 * 1024
    index_storage_quota_bytes: int = 50 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.request_secrets_by_key_id:
            raise ValueError("at least one request verification key is required")
        if len(set(self.approved_models)) != len(self.approved_models):
            raise ValueError("approved model IDs must be unique")
        if not isinstance(self.model_details, Mapping):
            raise ValueError("model details must be a mapping")
        normalized_model_details: dict[str, ModelDisplayDetails] = {}
        for model_id, details in self.model_details.items():
            if model_id not in self.approved_models:
                raise ValueError("model details must refer only to approved models")
            try:
                normalized_model_details[model_id] = (
                    details
                    if isinstance(details, ModelDisplayDetails)
                    else ModelDisplayDetails.model_validate(details)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"model details for {model_id!r} are invalid") from exc
        object.__setattr__(self, "model_details", normalized_model_details)
        if not self.index_schema_versions or len(set(self.index_schema_versions)) != len(self.index_schema_versions):
            raise ValueError("index schema versions must be nonempty and unique")
        if self.max_clock_skew_seconds < 1:
            raise ValueError("max clock skew must be positive")
        if self.max_request_bytes < 1024:
            raise ValueError("max request bytes is too small")
        if self.max_scope_entries < 1 or self.max_scope_entries > 256:
            raise ValueError("max scope entries must be between 1 and 256")
        if self.max_ingest_pages_per_batch < 1 or self.max_ingest_pages_per_batch > 250:
            raise ValueError("max ingest pages per batch must be between 1 and 250")
        if self.max_active_operations < 1:
            raise ValueError("max active operations must be positive")
        if self.executor_poll_seconds <= 0:
            raise ValueError("executor poll interval must be positive")
        if self.model_start_timeout_seconds < 1:
            raise ValueError("model startup timeout must be positive")
        if not isinstance(self.job_retention_seconds, int) or isinstance(self.job_retention_seconds, bool):
            raise ValueError("job retention must be an integer number of seconds")
        if self.job_retention_seconds < 1:
            raise ValueError("job retention must be positive")
        for name, value in (
            ("orphan grace", self.orphan_grace_seconds),
            ("document storage quota", self.document_storage_quota_bytes),
            ("index storage quota", self.index_storage_quota_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.orphan_grace_seconds > self.job_retention_seconds:
            raise ValueError("orphan grace must not exceed job retention")


class WorkerAPIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable


def _response_payload(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


def _job_view(job: StoredJob) -> JobView:
    error = None
    if job.error_code:
        error = JobError(
            code=job.error_code,
            message=job.error_message or "The operation failed.",
            retryable=job.retryable,
        )
    return JobView(
        job_id=job.job_id,
        operation=job.operation,
        state=job.state,
        created_at=job.created_at,
        updated_at=job.updated_at,
        progress=job.progress,
        result=job.result,
        error=error,
    )


def create_worker_app(
    config: WorkerConfig,
    *,
    model_controller: ModelController | None = None,
    compute_backend: ComputeBackend | None = None,
    model_readiness_probe: ModelReadinessProbe | None = None,
    clock: Callable[[], float] = time.time,
    auto_start_executor: bool = True,
) -> FastAPI:
    """Build a worker app intended to be bound only to a compute-host loopback port."""

    store = JobStore(config.state_database_path)
    startup_time = clock()
    store.recover_interrupted_operations(now=startup_time)
    store.purge_terminal_operations(
        environment=config.environment,
        updated_before=startup_time - config.job_retention_seconds,
    )
    controller = model_controller or UnavailableModelController()
    readiness_probe = model_readiness_probe or UnavailableModelReadinessProbe()
    state_root = (
        Path(config.state_directory)
        if config.state_directory is not None
        else Path(config.state_database_path).parent / "worker-runtime"
    )
    document_store = DocumentStore(
        state_root / "documents",
        max_bytes=config.document_storage_quota_bytes,
    )
    if compute_backend is None:
        # Kept local to avoid coupling the pure index module to any particular
        # network backend. Production runtime injects a validated backend.
        from .model_backend import UnavailableModelBackend

        compute_backend = UnavailableModelBackend()
    index_repository = IndexRepository(
        state_root / "indexes",
        compute_backend,
        max_bytes=config.index_storage_quota_bytes,
    )
    maintenance_lock = threading.Lock()

    def perform_storage_maintenance() -> dict[str, Any]:
        """Expire and prune derived state without racing a new submission."""

        with maintenance_lock:
            maintenance_time = clock()
            active_cutoff = maintenance_time - config.job_retention_seconds
            orphan_cutoff = maintenance_time - config.orphan_grace_seconds
            snapshot = store.storage_retention_snapshot(
                environment=config.environment,
                activated_before=active_cutoff,
            )
            if not snapshot.idle:
                return {"skipped": True, "reason": "active_operations"}
            documents_removed = document_store.prune(
                retained_finalized_scopes=snapshot.active_scopes,
                modified_before=orphan_cutoff,
            )
            indexes_removed = index_repository.prune(
                retained_artifact_ids=snapshot.active_artifact_ids,
                modified_before=orphan_cutoff,
            )
            return {
                "skipped": False,
                "expiredIndexes": snapshot.expired_indexes,
                "documentsRemoved": documents_removed,
                "indexesRemoved": indexes_removed,
            }

    # Start from a bounded state before accepting traffic. A failed retention
    # pass prevents startup rather than silently retaining sensitive data.
    perform_storage_maintenance()
    executor = WorkerExecutor(
        environment=config.environment,
        job_store=store,
        document_store=document_store,
        index_repository=index_repository,
        model_controller=controller,
        model_readiness_probe=readiness_probe,
        approved_models=config.approved_models,
        model_start_timeout_seconds=config.model_start_timeout_seconds,
        clock=clock,
        poll_seconds=config.executor_poll_seconds,
        idle_maintenance=perform_storage_maintenance,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if auto_start_executor:
            executor.start()
        try:
            yield
        finally:
            if auto_start_executor:
                executor.stop()

    app = FastAPI(
        title="Novo Chat Internal Worker",
        version=PROTOCOL_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.job_store = store
    app.state.worker_config = config
    app.state.model_controller = controller
    app.state.model_readiness_probe = readiness_probe
    app.state.document_store = document_store
    app.state.index_repository = index_repository
    app.state.executor = executor
    app.state.perform_storage_maintenance = perform_storage_maintenance

    def request_id_from(request: Request) -> str:
        verified = getattr(request.state, "verified_request", None)
        if verified is None:
            raise WorkerAPIError(401, "AUTHENTICATION_REQUIRED", "Request authentication failed.")
        return str(verified.request_id)

    def bind_request(request: Request, body_request_id: str) -> str:
        signed_request_id = request_id_from(request)
        if signed_request_id != body_request_id:
            raise WorkerAPIError(400, "REQUEST_ID_MISMATCH", "Signed and body request IDs do not match.")
        return signed_request_id

    def validate_scope(scope: Sequence[NotebookScope]) -> None:
        if len(scope) > config.max_scope_entries:
            raise WorkerAPIError(400, "SCOPE_TOO_LARGE", "Notebook scope exceeds the configured limit.")
        unsupported = [
            entry.index_schema_version
            for entry in scope
            if entry.index_schema_version not in config.index_schema_versions
        ]
        if unsupported:
            raise WorkerAPIError(400, "INDEX_SCHEMA_NOT_SUPPORTED", "Index schema version is not supported.")

    def require_approved_model(model: str) -> None:
        if model not in config.approved_models:
            raise WorkerAPIError(404, "MODEL_NOT_APPROVED", "Requested model is not available.")

    def submit_job(
        request_model: Any,
        *,
        operation: JobOperation,
        payload: Mapping[str, Any],
        dedupe_key: str | None = None,
    ) -> JobSubmissionResponse:
        validate_scope(request_model.scope)
        try:
            with maintenance_lock:
                store.purge_terminal_operations(
                    environment=config.environment,
                    updated_before=clock() - config.job_retention_seconds,
                )
                outcome = store.submit(
                    environment=config.environment,
                    actor_user_id=request_model.actor_user_id,
                    idempotency_key=request_model.idempotency_key,
                    request_id=request_model.request_id,
                    operation=operation,
                    scope=request_model.scope,
                    payload=payload,
                    dedupe_key=dedupe_key,
                    max_active_operations=config.max_active_operations,
                    now=clock(),
                )
        except IdempotencyConflict as exc:
            raise WorkerAPIError(
                409,
                "IDEMPOTENCY_CONFLICT",
                "Idempotency key was already used for a different request.",
            ) from exc
        except QueueFull as exc:
            raise WorkerAPIError(429, "QUEUE_FULL", "Worker queue is full.", retryable=True) from exc
        executor.notify()
        return JobSubmissionResponse(request_id=request_model.request_id, job=_job_view(outcome.job))

    def error_response(request_id: str, error: WorkerAPIError | ProtocolError) -> JSONResponse:
        response_model = ErrorResponse(
            request_id=request_id,
            error=JobError(code=error.code, message=error.message, retryable=error.retryable),
        )
        return JSONResponse(_response_payload(response_model), status_code=error.status_code)

    @app.middleware("http")
    async def authenticate_and_sign(request: Request, call_next: Callable[..., Any]) -> Response:
        candidate_request_id = request.headers.get("X-Novo-Chat-Request-Id", "")
        try:
            response_request_id = validate_request_id(candidate_request_id)
        except ValueError:
            response_request_id = new_request_id()

        response: Response | None = None
        declared_length: int | None = None
        if request.headers.getlist("transfer-encoding"):
            response = error_response(
                response_request_id,
                WorkerAPIError(
                    400,
                    "UNSUPPORTED_TRANSFER_ENCODING",
                    "Transfer-Encoding is not supported.",
                ),
            )
        else:
            content_lengths = request.headers.getlist("content-length")
            if len(content_lengths) > 1 or (content_lengths and "," in content_lengths[0]):
                response = error_response(
                    response_request_id,
                    WorkerAPIError(
                        400,
                        "AMBIGUOUS_REQUEST_FRAMING",
                        "Multiple Content-Length values are not accepted.",
                    ),
                )
            elif not content_lengths and request.method in {"POST", "PUT", "PATCH"}:
                response = error_response(
                    response_request_id,
                    WorkerAPIError(411, "LENGTH_REQUIRED", "Content-Length is required."),
                )
            elif content_lengths:
                raw_content_length = content_lengths[0].strip()
                if not re.fullmatch(r"[0-9]+", raw_content_length):
                    response = error_response(
                        response_request_id,
                        WorkerAPIError(400, "INVALID_CONTENT_LENGTH", "Content-Length header is invalid."),
                    )
                else:
                    normalized_length = raw_content_length.lstrip("0") or "0"
                    maximum_length = str(config.max_request_bytes)
                    if len(normalized_length) > len(maximum_length) or (
                        len(normalized_length) == len(maximum_length)
                        and normalized_length > maximum_length
                    ):
                        response = error_response(
                            response_request_id,
                            WorkerAPIError(
                                413,
                                "REQUEST_TOO_LARGE",
                                "Request body exceeds the configured limit.",
                            ),
                        )
                    else:
                        declared_length = int(normalized_length)
            else:
                declared_length = 0

        if response is None:
            chunks: list[bytes] = []
            total = 0
            try:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > config.max_request_bytes:
                        response = error_response(
                            response_request_id,
                            WorkerAPIError(
                                413,
                                "REQUEST_TOO_LARGE",
                                "Request body exceeds the configured limit.",
                            ),
                        )
                        break
                    chunks.append(chunk)
            except ClientDisconnect:
                response = error_response(
                    response_request_id,
                    WorkerAPIError(400, "REQUEST_BODY_INCOMPLETE", "Request body is incomplete."),
                )
            except Exception:
                response = error_response(
                    response_request_id,
                    WorkerAPIError(400, "REQUEST_BODY_INVALID", "Request body could not be read."),
                )

        if response is None:
            if total != declared_length:
                response = error_response(
                    response_request_id,
                    WorkerAPIError(
                        400,
                        "CONTENT_LENGTH_MISMATCH",
                        "Content-Length does not match the request body.",
                    ),
                )
            else:
                body = b"".join(chunks)
                request._body = body
                try:
                    verified = verify_request(
                        headers=request.headers,
                        secrets_by_key_id=config.request_secrets_by_key_id,
                        expected_environment=config.environment,
                        expected_audience=config.audience,
                        method=request.method,
                        path=request.url.path,
                        query=request.scope.get("query_string", b""),
                        body=body,
                        replay_guard=store,
                        now=int(clock()),
                        max_clock_skew_seconds=config.max_clock_skew_seconds,
                    )
                    request.state.verified_request = verified
                    response_request_id = verified.request_id
                    response = await call_next(request)
                except ProtocolError as exc:
                    response = error_response(response_request_id, exc)
                except Exception:
                    response = error_response(
                        response_request_id,
                        WorkerAPIError(500, "INTERNAL_ERROR", "The worker could not process the request.", retryable=True),
                    )

        response_body = b""
        if hasattr(response, "body_iterator"):
            async for chunk in response.body_iterator:
                response_body += chunk
        else:
            response_body = bytes(getattr(response, "body", b""))
        preserved_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower()
            not in {
                "content-length",
                "x-novo-chat-protocol",
                "x-novo-chat-audience",
                "x-novo-chat-environment",
                "x-novo-chat-key-id",
                "x-novo-chat-timestamp",
                "x-novo-chat-request-id",
                "x-novo-chat-status",
                "x-novo-chat-signature",
            }
        }
        preserved_headers["Cache-Control"] = "private, no-store"
        signed_headers = sign_response(
            secret=config.response_secret,
            key_id=config.response_key_id,
            environment=config.environment,
            audience=config.audience,
            request_id=response_request_id,
            status_code=response.status_code,
            body=response_body,
            timestamp=int(clock()),
        )
        preserved_headers.update(signed_headers)
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=preserved_headers,
            media_type=response.media_type,
            background=response.background,
        )

    @app.exception_handler(WorkerAPIError)
    async def handle_worker_error(request: Request, exc: WorkerAPIError) -> JSONResponse:
        return error_response(request_id_from(request), exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, _exc: RequestValidationError) -> JSONResponse:
        return error_response(
            request_id_from(request),
            WorkerAPIError(400, "INVALID_REQUEST", "Request body is invalid."),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        return error_response(
            request_id_from(request),
            WorkerAPIError(500, "INTERNAL_ERROR", "The worker could not process the request.", retryable=True),
        )

    @app.get(f"{INTERNAL_PREFIX}/health")
    def health(request: Request) -> dict[str, Any]:
        execution_healthy = (not auto_start_executor or executor.running) and executor.maintenance_healthy
        healthy = (
            store.ping()
            and document_store.healthy()
            and index_repository.healthy()
            and execution_healthy
        )
        active_count = store.active_operation_count(environment=config.environment) if healthy else config.max_active_operations
        response_model = HealthResponse(
            request_id=request_id_from(request),
            environment=config.environment,
            state=WorkerState.READY if healthy else WorkerState.DEGRADED,
            queue_healthy=healthy and active_count < config.max_active_operations,
            index_service_healthy=document_store.healthy() and index_repository.healthy(),
        )
        return _response_payload(response_model)

    @app.get(f"{INTERNAL_PREFIX}/capabilities")
    def capabilities(request: Request) -> dict[str, Any]:
        response_model = CapabilitiesResponse(
            request_id=request_id_from(request),
            operations=tuple(JobOperation),
            approved_models=config.approved_models,
            model_details=dict(config.model_details),
            index_schema_versions=config.index_schema_versions,
            max_scope_entries=config.max_scope_entries,
            max_ingest_pages_per_batch=config.max_ingest_pages_per_batch,
        )
        return _response_payload(response_model)

    @app.post(f"{INTERNAL_PREFIX}/indexes/status")
    def index_status(body: IndexStatusRequest, request: Request) -> dict[str, Any]:
        bind_request(request, body.request_id)
        validate_scope(body.scope)
        index_states = [
            index_repository.artifact_status(store.active_index(entry, environment=config.environment))
            for entry in body.scope
        ]
        response_model = IndexStatusResponse(
            request_id=body.request_id,
            indexes=tuple(
                IndexStatusItem(
                    notebook_id=entry.notebook_id,
                    content_revision=entry.content_revision,
                    index_schema_version=entry.index_schema_version,
                    exact_ready=status.exact_ready,
                    activated_at=status.activated_at,
                    chunk_count=status.chunk_count,
                )
                for entry, status in zip(body.scope, index_states)
            ),
        )
        return _response_payload(response_model)

    @app.post(f"{INTERNAL_PREFIX}/ingest/batches", status_code=202)
    def ingest_batch(body: IngestBatchRequest, request: Request) -> dict[str, Any]:
        bind_request(request, body.request_id)
        validate_scope(body.scope)
        if len(body.pages) > config.max_ingest_pages_per_batch:
            raise WorkerAPIError(400, "INGEST_BATCH_TOO_LARGE", "Ingest batch exceeds the configured page limit.")
        pages_payload = [page.model_dump(mode="json", by_alias=True) for page in body.pages]
        expected_checksum = ingest_pages_checksum(body.pages)
        if body.batch_checksum != expected_checksum:
            raise WorkerAPIError(400, "CHECKSUM_MISMATCH", "Ingest batch checksum does not match its content.")
        response_model = submit_job(
            body,
            operation=JobOperation.INGEST_BATCH,
            payload={
                "batchNumber": body.batch_number,
                "batchChecksum": body.batch_checksum,
                "pages": pages_payload,
            },
        )
        return _response_payload(response_model)

    @app.post(f"{INTERNAL_PREFIX}/ingest/finalize", status_code=202)
    def ingest_finalize(body: IngestFinalizeRequest, request: Request) -> dict[str, Any]:
        bind_request(request, body.request_id)
        response_model = submit_job(
            body,
            operation=JobOperation.INGEST_FINALIZE,
            payload={
                "documentChecksum": body.document_checksum,
                "pageCount": body.page_count,
                "batchCount": body.batch_count,
            },
        )
        return _response_payload(response_model)

    @app.post(f"{INTERNAL_PREFIX}/indexes/rebuild", status_code=202)
    def rebuild(body: RebuildRequest, request: Request) -> dict[str, Any]:
        bind_request(request, body.request_id)
        response_model = submit_job(
            body,
            operation=JobOperation.INDEX_REBUILD,
            payload={"force": body.force},
        )
        return _response_payload(response_model)

    @app.post(f"{INTERNAL_PREFIX}/query", status_code=202)
    def query(body: QueryRequest, request: Request) -> dict[str, Any]:
        bind_request(request, body.request_id)
        validate_scope(body.scope)
        require_approved_model(body.model)
        if not all(
            index_repository.artifact_ready(store.active_index(entry, environment=config.environment))
            for entry in body.scope
        ):
            raise WorkerAPIError(
                409,
                "INDEX_NOT_READY",
                "An exact requested notebook index is not ready.",
                retryable=True,
            )
        response_model = submit_job(
            body,
            operation=JobOperation.QUERY,
            payload={
                "question": body.question,
                "retrievalQuestion": body.retrieval_question,
                "model": body.model,
                "strategy": body.strategy.value,
                "maxSources": body.max_sources,
                "retrievalTopK": body.retrieval_top_k,
            },
        )
        return _response_payload(response_model)

    @app.get(f"{INTERNAL_PREFIX}/models/status")
    def models_status(request: Request) -> dict[str, Any]:
        models: list[ModelStatusItem] = []
        signed_request_id = request_id_from(request)
        for model in config.approved_models:
            try:
                controller_status = controller.status(
                    model,
                    request_id=signed_request_id,
                    actor_user_id="gateway-service",
                )
                raw_state = str(controller_status.get("state") or "").lower()
                if raw_state in {"running", "ready"}:
                    state = (
                        ModelRuntimeState.READY
                        if readiness_probe.is_ready(model)
                        else ModelRuntimeState.STARTING
                    )
                elif raw_state == "stopped":
                    state = ModelRuntimeState.STOPPED
                else:
                    state = ModelRuntimeState.FAILED
            except Exception:
                state = ModelRuntimeState.FAILED
            models.append(ModelStatusItem(model=model, state=state))
        response_model = ModelStatusResponse(request_id=signed_request_id, models=tuple(models))
        return _response_payload(response_model)

    def submit_model_job(body: ModelJobRequest, request: Request, operation: JobOperation) -> dict[str, Any]:
        bind_request(request, body.request_id)
        require_approved_model(body.model)
        response_model = submit_job(body, operation=operation, payload={"model": body.model})
        return _response_payload(response_model)

    @app.post(f"{INTERNAL_PREFIX}/models/start", status_code=202)
    def start_model(body: ModelJobRequest, request: Request) -> dict[str, Any]:
        return submit_model_job(body, request, JobOperation.MODEL_START)

    @app.post(f"{INTERNAL_PREFIX}/models/stop", status_code=202)
    def stop_model(body: ModelJobRequest, request: Request) -> dict[str, Any]:
        return submit_model_job(body, request, JobOperation.MODEL_STOP)

    @app.get(f"{INTERNAL_PREFIX}/jobs/{{job_id}}")
    def job_status(job_id: str, request: Request) -> dict[str, Any]:
        response_request_id = request_id_from(request)
        if not _JOB_ID.fullmatch(job_id):
            raise WorkerAPIError(404, "JOB_NOT_FOUND", "Job was not found.")
        try:
            job = store.get(job_id, environment=config.environment)
        except JobNotFound as exc:
            raise WorkerAPIError(404, "JOB_NOT_FOUND", "Job was not found.") from exc
        response_model = JobStatusResponse(request_id=response_request_id, job=_job_view(job))
        return _response_payload(response_model)

    return app


__all__ = ["INTERNAL_PREFIX", "WorkerConfig", "create_worker_app"]
