from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from novo_chat.compute import IndexRepository
from novo_chat.documents import DocumentStore
from novo_chat.executor import WorkerExecutor
from novo_chat.jobs import JobStore
from novo_chat.model_controller import UnixSocketModelController
from novo_chat.protocol import (
    GenerationResult,
    JobOperation,
    JobState,
    NotebookScope,
    PageDocument,
    QueryTimings,
    document_pages_checksum,
    ingest_pages_checksum,
)


class FakeBackend:
    def __init__(self) -> None:
        self.fail_revisions: set[str] = set()
        self.last_hits: list[dict[str, Any]] = []
        self.answer_override: str | None = None
        self.timings: QueryTimings | None = None

    @staticmethod
    def vector(text: str) -> np.ndarray:
        lowered = text.lower()
        return np.asarray(
            [
                1.0 + lowered.count("alpha") * 2.0,
                1.0 + lowered.count("beta") * 2.0,
                1.0 + lowered.count("gamma") * 2.0,
                1.0,
            ],
            dtype=np.float32,
        )

    def embed_documents(self, texts: Sequence[str], *, scope: NotebookScope, progress_callback=None) -> np.ndarray:
        if scope.content_revision in self.fail_revisions:
            raise RuntimeError("test-only embedding failure")
        if not texts:
            # Intentionally differs from populated artifacts' width; mixed
            # empty/nonempty queries must ignore this zero-row dimension.
            return np.empty((0, 9), dtype=np.float32)
        vectors = []
        for completed, text in enumerate(texts, start=1):
            vectors.append(self.vector(text))
            if progress_callback is not None:
                progress_callback(completed, len(texts))
        return np.vstack(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.vector(text)

    def generate(
        self,
        question: str,
        hits: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_sources: int,
    ) -> str | GenerationResult:
        self.last_hits = [dict(hit) for hit in hits]
        if self.answer_override is not None:
            return self.answer_override
        if not hits:
            return "The provided documents do not contain this information."
        answer = f"Grounded {question} [{hits[0]['source_idx']}]"
        if self.timings is not None:
            return GenerationResult(answer=answer, timings=self.timings)
        return answer


class FakeReadiness:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.waits: list[tuple[str, float]] = []
        self.progress_polls = 0

    def is_ready(self, model: str) -> bool:
        return self.ready

    def wait_until_ready(
        self,
        model: str,
        *,
        timeout_seconds: float,
        progress_callback=None,
    ) -> bool:
        self.waits.append((model, timeout_seconds))
        for _ in range(self.progress_polls):
            if progress_callback is not None:
                progress_callback()
        return self.ready


class RecordingController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []
        self.states: dict[str, str] = {}
        self.progresses: dict[str, list[float]] = {}

    def status(self, model: str, *, request_id: str, actor_user_id: str):
        self.calls.append(("status", model, request_id, actor_user_id))
        status = {"state": self.states.get(model, "stopped")}
        values = self.progresses.get(model, [])
        if values:
            status["progress"] = values.pop(0)
        return status

    def start(self, model: str, *, request_id: str, actor_user_id: str):
        self.calls.append(("start", model, request_id, actor_user_id))
        self.states[model] = "running"
        return {"state": "running", "changed": True}

    def stop(self, model: str, *, request_id: str, actor_user_id: str):
        self.calls.append(("stop", model, request_id, actor_user_id))
        self.states[model] = "stopped"
        return {"state": "stopped", "changed": True}


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 1_700_000_000.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class WorkerExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.clock = IncrementingClock()
        self.backend = FakeBackend()
        self.readiness = FakeReadiness()
        self.controller = RecordingController()
        self.store = JobStore(root / "worker.sqlite3")
        self.documents = DocumentStore(root / "documents")
        self.indexes = IndexRepository(root / "indexes", self.backend)
        self.executor = WorkerExecutor(
            environment="staging",
            job_store=self.store,
            document_store=self.documents,
            index_repository=self.indexes,
            model_controller=self.controller,
            model_readiness_probe=self.readiness,
            approved_models=("model:a",),
            model_start_timeout_seconds=12,
            clock=self.clock,
            poll_seconds=0.05,
        )

    @staticmethod
    def scope(notebook: str, revision: str) -> NotebookScope:
        return NotebookScope(
            notebook_id=notebook,
            content_revision=revision,
            index_schema_version="index-v1",
        )

    @staticmethod
    def page(page_id: str, text: str) -> PageDocument:
        return PageDocument(
            page_id=page_id,
            title=f"Title {page_id}",
            text=text,
            tags=("tag",),
            source_url=f"/?page={page_id}",
        )

    def submit(
        self,
        operation: JobOperation,
        scope: Sequence[NotebookScope],
        payload: Mapping[str, Any],
        *,
        actor: str = "user-a",
        idempotency_key: str | None = None,
        request_id: str | None = None,
    ):
        return self.store.submit(
            environment="staging",
            actor_user_id=actor,
            idempotency_key=idempotency_key or f"idem-{uuid4()}",
            request_id=request_id or str(uuid4()),
            operation=operation,
            scope=scope,
            payload=payload,
            now=self.clock(),
        )

    def execute(self, outcome):
        self.assertTrue(self.executor.run_once())
        return self.store.get(outcome.job.job_id, environment="staging")

    def ingest_and_finalize(self, scope: NotebookScope, pages: Sequence[PageDocument], batch_size: int = 1):
        batches = [tuple(pages[index : index + batch_size]) for index in range(0, len(pages), batch_size)]
        for batch_number, batch in enumerate(batches):
            outcome = self.submit(
                JobOperation.INGEST_BATCH,
                (scope,),
                {
                    "batchNumber": batch_number,
                    "batchChecksum": ingest_pages_checksum(batch),
                    "pages": [page.model_dump(mode="json", by_alias=True) for page in batch],
                },
            )
            self.assertEqual(self.execute(outcome).state, JobState.SUCCEEDED)
        finalize = self.submit(
            JobOperation.INGEST_FINALIZE,
            (scope,),
            {
                "documentChecksum": document_pages_checksum(pages),
                "pageCount": len(pages),
                "batchCount": len(batches),
            },
        )
        self.assertEqual(self.execute(finalize).state, JobState.SUCCEEDED)

    def rebuild(self, scopes: Sequence[NotebookScope]):
        outcome = self.submit(JobOperation.INDEX_REBUILD, scopes, {"force": False})
        return self.execute(outcome)

    def test_full_ingest_rebuild_query_pipeline_and_scope_filtering(self):
        alpha = self.scope("notebook-alpha", "rev-alpha")
        beta = self.scope("notebook-beta", "rev-beta")
        empty = self.scope("notebook-empty", "rev-empty")
        self.ingest_and_finalize(alpha, (self.page("page-alpha", "alpha protocol result"),))
        self.ingest_and_finalize(beta, (self.page("page-beta", "beta unrelated result"),))
        self.ingest_and_finalize(empty, ())
        rebuilt = self.rebuild((alpha, beta, empty))
        self.assertEqual(rebuilt.state, JobState.SUCCEEDED)
        self.assertTrue(all(self.store.index_status((alpha, beta, empty), environment="staging")))
        self.backend.timings = QueryTimings(prompt_eval_count=12_345, num_ctx=262_144, eval_count=345)

        query = self.submit(
            JobOperation.QUERY,
            (alpha, empty),
            {
                "question": "alpha?",
                "retrievalQuestion": "alpha",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 4,
            },
        )
        completed = self.execute(query)
        self.assertEqual(completed.state, JobState.SUCCEEDED)
        self.assertEqual(completed.result["kind"], "query")
        self.assertEqual(
            completed.result["timings"],
            {"prompt_eval_count": 12_345, "num_ctx": 262_144, "eval_count": 345},
        )
        self.assertEqual(completed.result["retrievalPlan"]["originalQuestion"], "alpha?")
        self.assertEqual(completed.result["retrievalPlan"]["mode"], "fallback")
        self.assertEqual(completed.progress_detail["stage"], "answering")
        self.assertEqual(
            completed.progress_detail["retrievalPlan"],
            completed.result["retrievalPlan"],
        )
        self.assertEqual({row["notebookId"] for row in completed.result["citations"]}, {alpha.notebook_id})
        self.assertEqual({row["notebook_id"] for row in self.backend.last_hits}, {alpha.notebook_id})

    def test_rebuild_reports_monotonic_progress_during_embedding(self):
        notebook = self.scope("notebook-progress", "rev-progress")
        pages = tuple(self.page(f"page-{position}", f"alpha result {position}") for position in range(4))
        self.ingest_and_finalize(notebook, pages)
        outcome = self.submit(JobOperation.INDEX_REBUILD, (notebook,), {"force": False})
        updates: list[float] = []
        original_set_progress = self.store.set_progress

        def record_progress(operation_id, progress, **kwargs):
            updates.append(float(progress))
            return original_set_progress(operation_id, progress, **kwargs)

        with patch.object(self.store, "set_progress", side_effect=record_progress):
            completed = self.execute(outcome)

        self.assertEqual(completed.state, JobState.SUCCEEDED)
        self.assertGreaterEqual(len([value for value in updates if 0.05 < value < 0.90]), 4)
        self.assertEqual(updates, sorted(updates))
        self.assertGreaterEqual(updates[-1], 0.99)

    def test_query_returns_ranked_hits_beyond_the_prompt_context(self):
        scope = self.scope("notebook-ranked", "rev-ranked")
        pages = tuple(
            self.page(f"page-{index}", f"alpha result {index} " + (chr(96 + index) * 1_500))
            for index in range(1, 5)
        )
        self.ingest_and_finalize(scope, pages, batch_size=4)
        self.assertEqual(self.rebuild((scope,)).state, JobState.SUCCEEDED)

        query = self.submit(
            JobOperation.QUERY,
            (scope,),
            {
                "question": "alpha?",
                "retrievalQuestion": "alpha",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 2,
                "retrievalTopK": 4,
            },
        )
        completed = self.execute(query)

        self.assertEqual(completed.state, JobState.SUCCEEDED)
        self.assertEqual(len(self.backend.last_hits), 2)
        citations = completed.result["citations"]
        self.assertEqual(len(citations), 4)
        self.assertEqual([row["sourceIdx"] for row in citations], [1, 2, 3, 4])
        self.assertEqual([row["usedInContext"] for row in citations], [True, True, False, False])
        self.assertTrue(all(row["file"].endswith(".md") for row in citations))
        self.assertTrue(all(row["chunkIdx"] == 0 for row in citations))
        self.assertTrue(all(isinstance(row["score"], float) for row in citations))
        self.assertTrue(all(isinstance(row["bm25"], float) for row in citations))
        self.assertTrue(all(isinstance(row["dense"], float) for row in citations))
        self.assertTrue(all(len(row["excerpt"]) > 1_000 for row in citations))

    def test_query_accepts_legacy_max_sources_above_ranked_retrieval_limit(self):
        scope = self.scope("notebook-legacy-depth", "rev-legacy-depth")
        self.ingest_and_finalize(scope, (self.page("page-alpha", "alpha result"),))
        self.assertEqual(self.rebuild((scope,)).state, JobState.SUCCEEDED)

        query = self.submit(
            JobOperation.QUERY,
            (scope,),
            {
                "question": "alpha?",
                "retrievalQuestion": "alpha",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 100,
                "retrievalTopK": 32,
            },
        )
        completed = self.execute(query)

        self.assertEqual(completed.state, JobState.SUCCEEDED)
        self.assertEqual(len(self.backend.last_hits), 1)

    def test_query_rejects_citation_to_ranked_hit_not_used_in_prompt(self):
        scope = self.scope("notebook-ranked", "rev-ranked")
        pages = tuple(
            self.page(f"page-{index}", f"alpha result {index}")
            for index in range(1, 5)
        )
        self.ingest_and_finalize(scope, pages, batch_size=4)
        self.assertEqual(self.rebuild((scope,)).state, JobState.SUCCEEDED)
        self.backend.answer_override = "This cites a ranked-only source [3]."

        query = self.submit(
            JobOperation.QUERY,
            (scope,),
            {
                "question": "alpha?",
                "retrievalQuestion": "alpha",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 2,
                "retrievalTopK": 4,
            },
        )
        failed = self.execute(query)

        self.assertEqual(len(self.backend.last_hits), 2)
        self.assertEqual(failed.state, JobState.FAILED)
        self.assertEqual(failed.error_code, "CITATION_VALIDATION_FAILED")

    def test_finalize_rejects_missing_extra_and_wrong_aggregate_checksum(self):
        scope = self.scope("notebook-a", "rev-a")
        page_a = self.page("page-a", "alpha")
        batch = self.submit(
            JobOperation.INGEST_BATCH,
            (scope,),
            {
                "batchNumber": 0,
                "batchChecksum": ingest_pages_checksum((page_a,)),
                "pages": [page_a.model_dump(mode="json", by_alias=True)],
            },
        )
        self.assertEqual(self.execute(batch).state, JobState.SUCCEEDED)
        missing = self.submit(
            JobOperation.INGEST_FINALIZE,
            (scope,),
            {
                "documentChecksum": document_pages_checksum((page_a,)),
                "pageCount": 1,
                "batchCount": 2,
            },
        )
        missing_job = self.execute(missing)
        self.assertEqual(missing_job.error_code, "INGEST_BATCHES_INCOMPLETE")

        wrong_checksum = self.submit(
            JobOperation.INGEST_FINALIZE,
            (scope,),
            {
                "documentChecksum": "sha256:" + "0" * 64,
                "pageCount": 1,
                "batchCount": 1,
            },
        )
        wrong_job = self.execute(wrong_checksum)
        self.assertEqual(wrong_job.error_code, "DOCUMENT_CHECKSUM_MISMATCH")

        page_b = self.page("page-b", "beta")
        extra_batch = self.submit(
            JobOperation.INGEST_BATCH,
            (scope,),
            {
                "batchNumber": 1,
                "batchChecksum": ingest_pages_checksum((page_b,)),
                "pages": [page_b.model_dump(mode="json", by_alias=True)],
            },
        )
        self.execute(extra_batch)
        extra = self.submit(
            JobOperation.INGEST_FINALIZE,
            (scope,),
            {
                "documentChecksum": document_pages_checksum((page_a,)),
                "pageCount": 1,
                "batchCount": 1,
            },
        )
        self.assertEqual(self.execute(extra).error_code, "INGEST_BATCHES_INCOMPLETE")

    def test_failed_rebuild_preserves_previous_active_index(self):
        old = self.scope("notebook-a", "rev-old")
        new = self.scope("notebook-a", "rev-new")
        self.ingest_and_finalize(old, (self.page("page-old", "alpha old"),))
        self.assertEqual(self.rebuild((old,)).state, JobState.SUCCEEDED)
        active_before = self.store.active_index(old, environment="staging")
        self.assertIsNotNone(active_before)

        self.ingest_and_finalize(new, (self.page("page-new", "alpha new"),))
        self.backend.fail_revisions.add("rev-new")
        failed = self.rebuild((new,))
        self.assertEqual(failed.state, JobState.FAILED)
        self.assertIsNone(self.store.active_index(new, environment="staging"))
        active_after = self.store.active_index(old, environment="staging")
        self.assertEqual(active_after.artifact_id, active_before.artifact_id)
        self.assertEqual(self.indexes.load_active(active_after).scope, old)

    def test_runner_rechecks_exact_revision_before_query(self):
        old = self.scope("notebook-a", "rev-old")
        new = self.scope("notebook-a", "rev-new")
        self.ingest_and_finalize(old, (self.page("page-old", "alpha"),))
        self.rebuild((old,))
        query = self.submit(
            JobOperation.QUERY,
            (old,),
            {
                "question": "alpha?",
                "retrievalQuestion": "alpha",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 4,
            },
        )
        # Simulate a newer version becoming active after submission but before
        # the query is claimed. It must not silently use the older artifact.
        self.store.activate_index(new, environment="staging", artifact_id="f" * 64, now=self.clock())
        failed = self.execute(query)
        self.assertEqual(failed.error_code, "INDEX_NOT_READY")

    def test_invalid_generated_citation_fails_query(self):
        scope = self.scope("notebook-a", "rev-a")
        self.ingest_and_finalize(scope, (self.page("page-a", "alpha result"),))
        self.rebuild((scope,))
        self.backend.answer_override = "Unsupported answer [99]"
        query = self.submit(
            JobOperation.QUERY,
            (scope,),
            {
                "question": "alpha?",
                "retrievalQuestion": "alpha",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 4,
            },
        )
        failed = self.execute(query)
        self.assertEqual(failed.state, JobState.FAILED)
        self.assertEqual(failed.error_code, "CITATION_VALIDATION_FAILED")

    def test_empty_index_requires_exact_no_information_answer(self):
        scope = self.scope("notebook-empty", "rev-empty")
        self.ingest_and_finalize(scope, ())
        self.rebuild((scope,))
        self.backend.answer_override = "Hallucinated answer without evidence."
        query = self.submit(
            JobOperation.QUERY,
            (scope,),
            {
                "question": "anything?",
                "retrievalQuestion": "anything",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 4,
            },
        )
        self.assertEqual(self.execute(query).error_code, "CITATION_VALIDATION_FAILED")

    def test_model_lifecycle_passes_audit_context_and_dedupes_only_while_active(self):
        request_one = str(uuid4())
        first = self.submit(
            JobOperation.MODEL_START,
            (),
            {"model": "model:a"},
            actor="user-a",
            idempotency_key="model-start-one",
            request_id=request_one,
        )
        first_job = self.execute(first)
        self.assertEqual(first_job.state, JobState.SUCCEEDED)
        retry = self.submit(
            JobOperation.MODEL_START,
            (),
            {"model": "model:a"},
            actor="user-a",
            idempotency_key="model-start-one",
            request_id=str(uuid4()),
        )
        self.assertEqual(retry.job.job_id, first.job.job_id)

        stop = self.submit(JobOperation.MODEL_STOP, (), {"model": "model:a"}, actor="user-a")
        self.assertEqual(self.execute(stop).state, JobState.SUCCEEDED)
        second_start = self.submit(JobOperation.MODEL_START, (), {"model": "model:a"}, actor="user-a")
        second_job = self.execute(second_start)
        self.assertEqual(second_job.state, JobState.SUCCEEDED)
        self.assertNotEqual(first.job.operation_id, second_start.job.operation_id)
        self.assertEqual(
            [call[0] for call in self.controller.calls],
            ["start", "stop", "start"],
        )
        self.assertEqual(self.controller.calls[0][2:], (request_one, "user-a"))
        self.assertEqual(self.readiness.waits, [("model:a", 12.0), ("model:a", 12.0)])

    def test_start_requires_backend_readiness(self):
        self.readiness.ready = False
        start = self.submit(JobOperation.MODEL_START, (), {"model": "model:a"})
        failed = self.execute(start)
        self.assertEqual(failed.state, JobState.FAILED)
        self.assertEqual(failed.error_code, "MODEL_READINESS_TIMEOUT")
        self.assertTrue(failed.retryable)

    def test_model_start_persists_each_controller_checkpoint_percentage(self):
        self.readiness.progress_polls = 4
        self.controller.progresses["model:a"] = [0.03, 0.49, 1.0, 0.98]
        start = self.submit(JobOperation.MODEL_START, (), {"model": "model:a"})

        with patch.object(self.store, "set_progress", wraps=self.store.set_progress) as updates:
            completed = self.execute(start)

        self.assertEqual(completed.state, JobState.SUCCEEDED)
        self.assertEqual(
            [call.args[1] for call in updates.call_args_list],
            [0.03, 0.49, 1.0],
        )

    def test_starting_second_model_stops_running_approved_model_first(self):
        self.controller.states["model:a"] = "running"
        switching_executor = WorkerExecutor(
            environment="staging",
            job_store=self.store,
            document_store=self.documents,
            index_repository=self.indexes,
            model_controller=self.controller,
            model_readiness_probe=self.readiness,
            approved_models=("model:a", "model:b"),
            model_start_timeout_seconds=12,
            clock=self.clock,
        )
        start = self.submit(JobOperation.MODEL_START, (), {"model": "model:b"})
        self.assertTrue(switching_executor.run_once())
        completed = self.store.get(start.job.job_id, environment="staging")
        self.assertEqual(completed.state, JobState.SUCCEEDED)
        self.assertEqual(
            [(action, model) for action, model, _request, _actor in self.controller.calls],
            [("status", "model:a"), ("stop", "model:a"), ("start", "model:b")],
        )

    def test_restart_requeues_safe_ingest_but_fails_running_query(self):
        scope = self.scope("notebook-a", "rev-a")
        page = self.page("page-a", "alpha")
        ingest = self.submit(
            JobOperation.INGEST_BATCH,
            (scope,),
            {
                "batchNumber": 0,
                "batchChecksum": ingest_pages_checksum((page,)),
                "pages": [page.model_dump(mode="json", by_alias=True)],
            },
        )
        claimed = self.store.claim_next_operation(environment="staging", now=self.clock())
        self.assertEqual(claimed.operation_id, ingest.job.operation_id)
        self.store.recover_interrupted_operations(now=self.clock())
        self.assertEqual(self.execute(ingest).state, JobState.SUCCEEDED)

        query = self.submit(
            JobOperation.QUERY,
            (scope,),
            {
                "question": "alpha?",
                "retrievalQuestion": "alpha",
                "model": "model:a",
                "strategy": "hybrid",
                "maxSources": 4,
            },
        )
        claimed_query = self.store.claim_next_operation(environment="staging", now=self.clock())
        self.assertEqual(claimed_query.operation_id, query.job.operation_id)
        self.store.recover_interrupted_operations(now=self.clock())
        self.assertEqual(self.store.get(query.job.job_id, environment="staging").error_code, "WORKER_RESTARTED")


class UnixSocketControllerClientTests(unittest.TestCase):
    def test_client_matches_json_line_controller_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "modelctl.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path))
            listener.listen(1)
            captured: dict[str, Any] = {}

            def serve() -> None:
                connection, _ = listener.accept()
                with connection:
                    raw = b""
                    while not raw.endswith(b"\n"):
                        raw += connection.recv(4096)
                    captured.update(json.loads(raw))
                    response = {
                        "ok": True,
                        "version": 1,
                        "requestId": captured["requestId"],
                        "model": captured["model"],
                        "action": captured["action"],
                        "status": {"state": "running", "changed": True, "progress": 0.49},
                        "statusCode": 200,
                    }
                    connection.sendall(json.dumps(response).encode() + b"\n")
                listener.close()

            thread = threading.Thread(target=serve)
            thread.start()
            request_id = str(uuid4())
            client = UnixSocketModelController(path, timeout_seconds=2)
            status = client.start("model:a", request_id=request_id, actor_user_id="user-a")
            thread.join(timeout=2)
            self.assertEqual(status, {"state": "running", "changed": True, "progress": 0.49})
            self.assertEqual(
                captured,
                {
                    "version": 1,
                    "requestId": request_id,
                    "actorId": "user-a",
                    "action": "start",
                    "model": "model:a",
                },
            )


if __name__ == "__main__":
    unittest.main()
