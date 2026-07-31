from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from novo_chat.gateway.app import create_app
from novo_chat.gateway.config import GatewaySettings
from novo_chat.gateway.novo_client import (
    NovoContext,
    NovoExportPage,
    NovoPagesResponse,
    NovoRevisionChanged,
)
from novo_chat.gateway.secrets import write_test_secret
from novo_chat.protocol import PageDocument, canonical_json, document_pages_checksum


def make_context(revision: str = "sha256:revision-a") -> NovoContext:
    return NovoContext.model_validate(
        {
            "apiVersion": "1",
            "novoVersion": "test",
            "user": {"id": "alice", "email": "alice@example.org", "firstName": "Alice", "role": "member"},
            "notebooks": [
                {
                    "id": "notebook-a",
                    "name": "Notebook A",
                    "accessRole": "editor",
                    "contentRevision": revision,
                    "pageCount": 2,
                }
            ],
        }
    )


def exported_page(page_id: str) -> dict[str, Any]:
    return {
        "id": page_id,
        "title": f"Title {page_id}",
        "text": f"Normalized {page_id}",
        "tags": ["tag"],
        "attachments": [],
        "sourceUrl": f"/?page={page_id}",
    }


class SyncNovoClient:
    def __init__(
        self,
        *,
        revision_change_once: bool = False,
        empty: bool = False,
        single_response_pages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.current = make_context()
        self.revision_change_once = revision_change_once
        self.empty = empty
        self.single_response_pages = single_response_pages
        self.export_calls: list[dict[str, Any]] = []
        self.context_calls = 0

    async def context(self, session_value: str) -> NovoContext:
        assert session_value == "alice-session"
        self.context_calls += 1
        return self.current

    async def export_pages(
        self,
        session_value: str,
        *,
        notebook_id: str,
        content_revision: str,
        cursor: str | None = None,
        limit: int = 100,
    ) -> NovoPagesResponse:
        assert session_value == "alice-session"
        self.export_calls.append(
            {
                "notebook_id": notebook_id,
                "content_revision": content_revision,
                "cursor": cursor,
                "limit": limit,
            }
        )
        if self.revision_change_once:
            self.revision_change_once = False
            self.current = make_context("sha256:revision-b")
            raise NovoRevisionChanged(notebook_id, content_revision)
        page_id = "page-a" if cursor is None else "page-b"
        if self.single_response_pages is not None:
            pages = self.single_response_pages
            complete = True
        else:
            pages = [] if self.empty else [exported_page(page_id)]
            complete = self.empty or cursor is not None
        return NovoPagesResponse.model_validate(
            {
                "apiVersion": "1",
                "notebook": {
                    "id": notebook_id,
                    "name": "Notebook A",
                    "contentRevision": content_revision,
                },
                "pages": pages,
                "nextCursor": None if complete else "cursor-2",
                "complete": complete,
            }
        )


class SyncWorkerClient:
    def __init__(self, *, queued_ingest: bool = False) -> None:
        self.ready: set[tuple[str, str, str]] = set()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.job_counter = 0
        self.queued_ingest = queued_ingest

    async def health(self) -> dict[str, Any]:
        return {"ok": True}

    async def capabilities(self) -> dict[str, Any]:
        return {"models": ["approved-model"]}

    async def model_status(self) -> dict[str, Any]:
        return {"models": {"approved-model": {"state": "ready", "healthy": True}}}

    async def index_status(self, **kwargs: Any) -> dict[str, Any]:
        scope = kwargs["scope"]
        self.events.append(("index_status", kwargs))
        return {
            "requestId": kwargs["request_id"],
            "indexes": [
                {
                    **row,
                    "exactReady": (
                        row["notebookId"], row["contentRevision"], row["indexSchemaVersion"]
                    ) in self.ready,
                }
                for row in scope
            ],
        }

    def _job(self, operation: str, state: str = "succeeded") -> dict[str, Any]:
        self.job_counter += 1
        job = {
            "jobId": f"job-{self.job_counter}",
            "operation": operation,
            "state": state,
            "createdAt": "2026-07-31T12:00:00Z",
            "updatedAt": "2026-07-31T12:00:01Z",
            "progress": 1.0 if state == "succeeded" else 0.0,
            "result": {"kind": "index", "indexes": [], "chunkCount": 0} if state == "succeeded" else None,
        }
        self.jobs[job["jobId"]] = job
        return {"protocolVersion": "1", "requestId": "mock", "job": job}

    async def ingest_batch(self, **kwargs: Any) -> dict[str, Any]:
        assert "cookie" not in kwargs and "session" not in kwargs
        self.events.append(("ingest_batch", kwargs))
        return self._job("ingest_batch", "queued" if self.queued_ingest else "succeeded")

    async def ingest_finalize(self, **kwargs: Any) -> dict[str, Any]:
        assert "cookie" not in kwargs and "session" not in kwargs
        self.events.append(("ingest_finalize", kwargs))
        return self._job("ingest_finalize")

    async def submit(self, **kwargs: Any) -> dict[str, Any]:
        assert "cookie" not in kwargs and "session" not in kwargs
        self.events.append((kwargs["operation"], kwargs))
        if kwargs["operation"] == "index_rebuild" and kwargs["idempotency_key"].startswith("sync-"):
            for row in kwargs["scope"]:
                self.ready.add((row["notebookId"], row["contentRevision"], row["indexSchemaVersion"]))
            return self._job("index_rebuild")
        return self._job(kwargs["operation"], "queued")

    async def job_status(self, job_id: str) -> dict[str, Any]:
        return {"protocolVersion": "1", "requestId": "mock", "job": self.jobs[job_id]}


def settings_for(tmp_path: Path, **overrides: Any) -> GatewaySettings:
    integration_secret = tmp_path / "integration-secret"
    worker_secret = tmp_path / "worker-secret"
    write_test_secret(integration_secret, "integration-test-secret-0123456789")
    write_test_secret(worker_secret, "worker-test-secret-01234567890123")
    values = {
        "base_path": "/chat",
        "novo_integration_secret_file": integration_secret,
        "worker_signing_secret_file": worker_secret,
        "job_db_path": tmp_path / "jobs.sqlite3",
        "worker_job_timeout_s": 1.0,
        "worker_poll_interval_s": 0.001,
        "novo_export_page_limit": 1,
        "novo_export_max_pages": 10,
    }
    values.update(overrides)
    return GatewaySettings(**values)


def request_headers() -> dict[str, str]:
    return {
        "cookie": "eln_session=alice-session",
        "origin": "http://testserver",
        "content-type": "application/json",
        "x-csrf-token": "placeholder",
        "idempotency-key": "browser-request-0001",
    }


def csrf_headers(client: TestClient) -> dict[str, str]:
    context = client.get("/chat/api/context", headers={"cookie": "eln_session=alice-session"}).json()
    return {**request_headers(), "x-csrf-token": context["csrfToken"]}


def test_ask_auto_synchronizes_missing_exact_index_before_query(tmp_path: Path) -> None:
    novo = SyncNovoClient()
    worker = SyncWorkerClient()
    app = create_app(settings_for(tmp_path), novo_client=novo, worker_client=worker)
    with TestClient(app) as client:
        response = client.post(
            "/chat/api/jobs",
            headers=csrf_headers(client),
            json={
                "operation": "ask",
                "corpus": "novo:notebook-a",
                "question": "What happened?",
                "model": "approved-model",
            },
        )
    assert response.status_code == 202
    assert response.json()["operation"] == "query"
    event_names = [name for name, _payload in worker.events]
    assert event_names == [
        "index_status",
        "ingest_batch",
        "ingest_batch",
        "ingest_finalize",
        "index_rebuild",
        "index_status",
        "query",
    ]
    assert [call["cursor"] for call in novo.export_calls] == [None, "cursor-2"]
    finalized = next(payload for name, payload in worker.events if name == "ingest_finalize")
    # Use the exact worker documents to verify the streaming aggregate checksum.
    documents = [
        PageDocument.model_validate(
            {
                "pageId": page_id,
                "title": f"Title {page_id}",
                "text": f"Normalized {page_id}",
                "tags": ["tag"],
                "attachments": [],
                "sourceUrl": f"/?page={page_id}",
            }
        )
        for page_id in ("page-a", "page-b")
    ]
    assert finalized["document_checksum"] == document_pages_checksum(documents)
    assert finalized["page_count"] == 2
    assert finalized["batch_count"] == 2


def test_explicit_rebuild_prepares_documents_but_returns_rebuild_job(tmp_path: Path) -> None:
    novo = SyncNovoClient()
    worker = SyncWorkerClient()
    app = create_app(settings_for(tmp_path), novo_client=novo, worker_client=worker)
    with TestClient(app) as client:
        response = client.post(
            "/chat/api/jobs",
            headers=csrf_headers(client),
            json={"operation": "index_rebuild", "corpus": "novo:notebook-a", "force": True},
        )
    assert response.status_code == 202
    assert response.json()["operation"] == "index_rebuild"
    rebuilds = [payload for name, payload in worker.events if name == "index_rebuild"]
    assert len(rebuilds) == 1
    assert rebuilds[0]["idempotency_key"] == "browser-request-0001"
    assert worker.jobs[response.json()["jobId"]]["state"] == "queued"


def test_revision_change_reloads_context_and_synchronizes_new_revision(tmp_path: Path) -> None:
    novo = SyncNovoClient(revision_change_once=True)
    worker = SyncWorkerClient()
    app = create_app(settings_for(tmp_path), novo_client=novo, worker_client=worker)
    with TestClient(app) as client:
        response = client.post(
            "/chat/api/jobs",
            headers=csrf_headers(client),
            json={
                "operation": "ask",
                "corpus": "novo:notebook-a",
                "question": "What changed?",
                "model": "approved-model",
            },
        )
    assert response.status_code == 202
    query = [payload for name, payload in worker.events if name == "query"][-1]
    assert query["scope"][0]["contentRevision"] == "sha256:revision-b"
    assert novo.context_calls >= 3


def test_empty_notebook_finalizes_without_an_ingest_batch(tmp_path: Path) -> None:
    novo = SyncNovoClient(empty=True)
    worker = SyncWorkerClient()
    app = create_app(settings_for(tmp_path), novo_client=novo, worker_client=worker)
    with TestClient(app) as client:
        response = client.post(
            "/chat/api/jobs",
            headers=csrf_headers(client),
            json={
                "operation": "ask",
                "corpus": "novo:notebook-a",
                "question": "Is this empty?",
                "model": "approved-model",
            },
        )
    assert response.status_code == 202
    assert not [payload for name, payload in worker.events if name == "ingest_batch"]
    finalized = next(payload for name, payload in worker.events if name == "ingest_finalize")
    assert finalized["page_count"] == 0
    assert finalized["batch_count"] == 0
    assert finalized["document_checksum"] == document_pages_checksum([])


def worker_page(export: dict[str, Any]) -> dict[str, Any]:
    page = NovoExportPage.model_validate(export)
    return PageDocument.model_validate(page.worker_document()).model_dump(mode="json", by_alias=True)


def test_export_response_is_partitioned_by_canonical_utf8_bytes(tmp_path: Path) -> None:
    exported = [
        {**exported_page("page-a"), "text": "alpha " + "é" * 80},
        {**exported_page("page-b"), "text": "beta " + "界" * 80},
    ]
    worker_pages = [worker_page(page) for page in exported]
    one_page_cap = max(len(canonical_json([page])) for page in worker_pages)
    assert len(canonical_json(worker_pages)) > one_page_cap
    novo = SyncNovoClient(single_response_pages=exported)
    worker = SyncWorkerClient()
    app = create_app(
        settings_for(
            tmp_path,
            novo_export_page_limit=8,
            ingest_batch_max_bytes=one_page_cap,
        ),
        novo_client=novo,
        worker_client=worker,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/api/jobs",
            headers=csrf_headers(client),
            json={"operation": "index_rebuild", "corpus": "novo:notebook-a", "force": True},
        )

    assert response.status_code == 202
    batches = [payload for name, payload in worker.events if name == "ingest_batch"]
    assert [payload["batch_number"] for payload in batches] == [0, 1]
    assert [len(payload["pages"]) for payload in batches] == [1, 1]
    assert all(len(canonical_json(payload["pages"])) <= one_page_cap for payload in batches)
    finalized = next(payload for name, payload in worker.events if name == "ingest_finalize")
    assert finalized["batch_count"] == 2
    assert finalized["document_checksum"] == document_pages_checksum(worker_pages)


def test_single_page_over_byte_cap_fails_safely_without_ingest(tmp_path: Path) -> None:
    exported = [{**exported_page("page-a"), "text": "界" * 80}]
    page_bytes = len(canonical_json([worker_page(exported[0])]))
    novo = SyncNovoClient(single_response_pages=exported)
    worker = SyncWorkerClient()
    app = create_app(
        settings_for(tmp_path, ingest_batch_max_bytes=page_bytes - 1),
        novo_client=novo,
        worker_client=worker,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat/api/jobs",
            headers=csrf_headers(client),
            json={"operation": "index_rebuild", "corpus": "novo:notebook-a", "force": True},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "This notebook is too large for the current Chat export limits."}
    assert not [payload for name, payload in worker.events if name == "ingest_batch"]


def test_explicit_rebuild_timeout_is_sanitized_504(tmp_path: Path) -> None:
    novo = SyncNovoClient()
    worker = SyncWorkerClient(queued_ingest=True)
    app = create_app(
        settings_for(tmp_path, worker_job_timeout_s=0.01, worker_poll_interval_s=0.1),
        novo_client=novo,
        worker_client=worker,
    )
    with TestClient(app) as client:
        response = client.post(
            "/chat/api/jobs",
            headers=csrf_headers(client),
            json={"operation": "index_rebuild", "corpus": "novo:notebook-a", "force": True},
        )
    assert response.status_code == 504
    assert response.json() == {"detail": "Notebook preparation timed out. Try again."}
