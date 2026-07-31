from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import novo_chat.gateway.app as gateway_app_module
from novo_chat.gateway.app import create_app
from novo_chat.gateway.config import GatewaySettings
from novo_chat.gateway.models import GatewayJobRequest
from novo_chat.gateway.novo_client import NovoContext, NovoUnauthenticated
from novo_chat.gateway.secrets import write_test_secret
from novo_chat.gateway.worker_client import WorkerRejected, WorkerUnavailable


def context_for(
    user_id: str,
    notebook_ids: tuple[str, ...] = ("notebook-a",),
    *,
    content_revisions: dict[str, str] | None = None,
) -> NovoContext:
    return NovoContext.model_validate(
        {
            "apiVersion": "1",
            "novoVersion": "test",
            "user": {
                "id": user_id,
                "email": f"{user_id}@example.org",
                "firstName": user_id.title(),
                "lastName": "User",
                "role": "member",
            },
            "notebooks": [
                {
                    "id": notebook_id,
                    "name": f"Notebook {notebook_id}",
                    "accessRole": "viewer",
                    "contentRevision": (content_revisions or {}).get(notebook_id, f"sha256:{notebook_id}"),
                    "updatedAt": "2026-07-31T12:00:00Z",
                    "pageCount": 3,
                    "attachmentCount": 1,
                    "textChars": 1200,
                }
                for notebook_id in notebook_ids
            ],
        }
    )


class MockNovoClient:
    def __init__(self) -> None:
        self.contexts: dict[str, NovoContext] = {
            "alice-session": context_for("alice"),
            "bob-session": context_for("bob"),
        }
        self.seen_sessions: list[str] = []

    async def context(self, session_value: str) -> NovoContext:
        self.seen_sessions.append(session_value)
        try:
            return self.contexts[session_value]
        except KeyError as exc:
            raise NovoUnauthenticated from exc


class MockWorkerClient:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.submissions: list[dict[str, Any]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.indexes_ready = True
        self.index_status_rows: dict[str, dict[str, Any]] = {}
        self.index_status_requests: list[dict[str, Any]] = []
        self.on_index_status: Any | None = None

    async def health(self) -> dict[str, Any]:
        if not self.available:
            raise WorkerUnavailable
        return {"ok": True}

    async def capabilities(self) -> dict[str, Any]:
        return {"models": [{"id": "approved-model"}]}

    async def model_status(self) -> dict[str, Any]:
        return {"models": {"approved-model": {"state": "running", "healthy": True}}}

    async def index_status(self, *, request_id: str, actor_user_id: str, scope: list[dict[str, str]]) -> dict[str, Any]:
        self.index_status_requests.append(
            {"requestId": request_id, "actorUserId": actor_user_id, "scope": scope}
        )
        if self.on_index_status is not None:
            self.on_index_status()
        return {
            "requestId": request_id,
            "indexes": [
                {
                    **row,
                    "exactReady": self.index_status_rows.get(row["notebookId"], {}).get(
                        "exactReady", self.indexes_ready
                    ),
                    **{
                        key: value
                        for key, value in self.index_status_rows.get(row["notebookId"], {}).items()
                        if key != "exactReady"
                    },
                }
                for row in scope
            ],
        }

    async def submit(self, **kwargs: Any) -> dict[str, Any]:
        if not self.available:
            raise WorkerUnavailable
        self.submissions.append(kwargs)
        job_id = f"job-{len(self.submissions)}"
        self.jobs[job_id] = {
            "jobId": job_id,
            "state": "completed",
            "operation": kwargs["operation"],
            "createdAt": "2026-07-31T12:00:00Z",
            "result": {
                "answer": "Grounded answer [1]",
                "hits": [{"notebookId": row["notebookId"], "source_idx": 1} for row in kwargs["scope"][:1]],
            },
        }
        return {"protocolVersion": "1", "requestId": kwargs["request_id"], "job": self.jobs[job_id]}

    async def job_status(self, job_id: str) -> dict[str, Any]:
        return {"protocolVersion": "1", "requestId": "status", "job": self.jobs[job_id]}


@pytest.fixture
def gateway(tmp_path: Path):
    integration_secret = tmp_path / "integration-secret"
    worker_secret = tmp_path / "worker-secret"
    write_test_secret(integration_secret, "integration-test-secret-0123456789")
    write_test_secret(worker_secret, "worker-test-secret-01234567890123")
    settings = GatewaySettings(
        base_path="/lab-chat",
        novo_api_base_url="http://127.0.0.1:3148/api/integrations/v1",
        novo_integration_secret_file=integration_secret,
        worker_base_url="http://127.0.0.1:18095/internal/v1",
        worker_signing_secret_file=worker_secret,
        job_db_path=tmp_path / "jobs.sqlite3",
    )
    novo = MockNovoClient()
    worker = MockWorkerClient()
    app = create_app(settings, novo_client=novo, worker_client=worker)
    with TestClient(app) as client:
        yield client, novo, worker


def session_headers(session: str = "alice-session") -> dict[str, str]:
    return {"cookie": f"eln_session={session}"}


def auth_headers(
    csrf: str,
    session: str = "alice-session",
    idempotency_key: str = "browser-request-0001",
) -> dict[str, str]:
    return {
        **session_headers(session),
        "origin": "http://testserver",
        "x-csrf-token": csrf,
        "content-type": "application/json",
        "idempotency-key": idempotency_key,
    }


async def fail_if_request_body_is_read(_request):
    raise AssertionError("request body was read before authentication and CSRF checks")
    yield b""


def test_unauthenticated_entry_redirects_to_novo_login(gateway) -> None:
    client, _novo, _worker = gateway
    response = client.get("/lab-chat/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/?returnTo=%2Flab-chat%2F"

    api_response = client.get("/lab-chat/api/context")
    assert api_response.status_code == 401
    assert api_response.json()["loginUrl"] == "/?returnTo=%2Flab-chat%2F"


def test_unauthenticated_oversized_job_body_is_not_read(gateway) -> None:
    client, _novo, worker = gateway
    with patch("starlette.requests.Request.stream", new=fail_if_request_body_is_read):
        response = client.post(
            "/lab-chat/api/jobs",
            content=b"x" * (64 * 1024 + 1),
            headers={"content-type": "application/json", "origin": "http://testserver"},
        )
    assert response.status_code == 401
    assert worker.submissions == []


def test_invalid_csrf_oversized_job_body_is_not_read(gateway) -> None:
    client, _novo, worker = gateway
    with patch("starlette.requests.Request.stream", new=fail_if_request_body_is_read):
        response = client.post(
            "/lab-chat/api/jobs",
            content=b"x" * (64 * 1024 + 1),
            headers=auth_headers("invalid-csrf-token"),
        )
    assert response.status_code == 403
    assert worker.submissions == []


def test_authenticated_oversized_job_body_is_rejected(gateway) -> None:
    client, _novo, worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    response = client.post(
        "/lab-chat/api/jobs",
        content=b"x" * (64 * 1024 + 1),
        headers=auth_headers(context["csrfToken"]),
    )
    assert response.status_code == 413
    assert worker.submissions == []


def test_oversized_corpus_identifier_is_rejected(gateway) -> None:
    client, _novo, worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    response = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={"operation": "index_rebuild", "corpus": f"novo:{'a' * 193}"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Request body is invalid"}
    assert worker.submissions == []


def test_configurable_base_path_serves_relative_ui_assets(gateway) -> None:
    client, _novo, _worker = gateway
    response = client.get("/lab-chat/", headers=session_headers())
    assert response.status_code == 200
    assert 'content="/lab-chat"' in response.text
    assert 'href="static/favicon.ico"' in response.text
    assert 'href="static/favicon.png"' in response.text
    assert 'href="static/styles.css"' in response.text
    assert 'src="static/app.js"' in response.text
    assert 'href="/static' not in response.text
    assert 'src="/static' not in response.text
    assert client.get("/lab-chat/static/app.js").status_code == 200
    assert client.get("/lab-chat/static/index.html").status_code == 404


def test_favicon_routes_are_explicit_and_base_relative(gateway, monkeypatch, tmp_path: Path) -> None:
    client, _novo, _worker = gateway
    asset_directory = tmp_path / "approved-assets"
    asset_directory.mkdir()
    (asset_directory / "favicon.ico").write_bytes(b"approved-ico")
    (asset_directory / "favicon.png").write_bytes(b"approved-png")
    monkeypatch.setattr(gateway_app_module, "WEB_DIR", asset_directory)

    ico = client.get("/lab-chat/static/favicon.ico")
    png = client.get("/lab-chat/static/favicon.png")

    assert ico.status_code == 200
    assert ico.headers["content-type"].startswith("image/x-icon")
    assert ico.content == b"approved-ico"
    assert png.status_code == 200
    assert png.headers["content-type"].startswith("image/png")
    assert png.content == b"approved-png"
    assert client.get("/favicon.ico").status_code == 404
    assert client.get("/lab-chat/favicon.ico").status_code == 404


def test_index_status_requires_auth_and_never_calls_worker_for_an_inaccessible_corpus(gateway) -> None:
    client, _novo, worker = gateway

    unauthenticated = client.get("/lab-chat/api/index-status?corpus=novo:notebook-a")
    inaccessible = client.get(
        "/lab-chat/api/index-status?corpus=novo:notebook-other",
        headers=session_headers(),
    )

    assert unauthenticated.status_code == 401
    assert inaccessible.status_code == 404
    assert inaccessible.json() == {"detail": "Corpus not found"}
    assert worker.index_status_requests == []


def test_index_status_returns_only_authorized_exact_scope_and_safe_metadata(gateway) -> None:
    client, novo, worker = gateway
    novo.contexts["alice-session"] = context_for("alice", ("notebook-b", "notebook-a"))
    worker.index_status_rows = {
        "notebook-a": {
            "exactReady": True,
            "activatedAt": "2026-07-31T20:00:00Z",
            "chunkCount": 12,
            "internalArtifactId": "must-not-leak",
        },
        "notebook-b": {"exactReady": False},
    }

    response = client.get(
        "/lab-chat/api/index-status?corpus=novo:all",
        headers=session_headers(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "corpus": "novo:all",
        "exactReady": False,
        "indexes": [
            {
                "notebookId": "notebook-a",
                "notebookName": "Notebook notebook-a",
                "contentRevision": "sha256:notebook-a",
                "exactReady": True,
                "activatedAt": "2026-07-31T20:00:00Z",
                "chunkCount": 12,
            },
            {
                "notebookId": "notebook-b",
                "notebookName": "Notebook notebook-b",
                "contentRevision": "sha256:notebook-b",
                "exactReady": False,
            },
        ],
    }
    assert worker.index_status_requests[0]["actorUserId"] == "alice"
    assert [row["notebookId"] for row in worker.index_status_requests[0]["scope"]] == [
        "notebook-a",
        "notebook-b",
    ]
    assert "internalArtifactId" not in response.text

    worker.index_status_requests.clear()
    selected = client.get(
        "/lab-chat/api/index-status?corpus=novo:notebook-b",
        headers=session_headers(),
    )
    assert selected.status_code == 200
    assert [row["notebookId"] for row in selected.json()["indexes"]] == ["notebook-b"]
    assert [row["notebookId"] for row in worker.index_status_requests[0]["scope"]] == ["notebook-b"]


def test_index_status_rechecks_live_revision_before_releasing_metadata(gateway) -> None:
    client, novo, worker = gateway

    def change_revision() -> None:
        novo.contexts["alice-session"] = context_for(
            "alice",
            content_revisions={"notebook-a": "sha256:changed"},
        )

    worker.on_index_status = change_revision
    response = client.get(
        "/lab-chat/api/index-status?corpus=novo:notebook-a",
        headers=session_headers(),
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Novo notebook content or access changed during the status check. Try again."
    }


def test_context_uses_live_novo_session_and_degrades_independently(gateway) -> None:
    client, novo, worker = gateway
    response = client.get("/lab-chat/api/context", headers=session_headers())
    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["id"] == "alice"
    assert [row["corpus_key"] for row in payload["corpora"]] == ["novo:all", "novo:notebook-a"]
    assert payload["corpora"][0]["name"] == "All notebooks"
    assert payload["corpora"][0]["updated_at"] == "2026-07-31T12:00:00Z"
    assert payload["worker"]["available"] is True
    assert novo.seen_sessions[-1] == "alice-session"

    worker.available = False
    degraded = client.get("/lab-chat/api/context", headers=session_headers()).json()
    assert degraded["worker"]["available"] is False
    assert "Novo is still available" in degraded["worker"]["detail"]


def test_context_accepts_typed_worker_health_without_synthetic_ok(gateway) -> None:
    client, _novo, worker = gateway

    async def typed_health() -> dict[str, Any]:
        return {"state": "ready", "queueHealthy": True, "indexServiceHealthy": True}

    worker.health = typed_health
    response = client.get("/lab-chat/api/context", headers=session_headers())
    assert response.status_code == 200
    assert response.json()["worker"]["available"] is True


def test_submit_requires_same_origin_json_and_csrf(gateway) -> None:
    client, _novo, worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    body = {
        "operation": "ask",
        "corpus": "novo:notebook-a",
        "question": "What happened?",
        "model": "approved-model",
    }
    assert client.post("/lab-chat/api/jobs", headers=session_headers(), json=body).status_code == 403
    assert client.post(
        "/lab-chat/api/jobs",
        headers={**auth_headers(context["csrfToken"]), "origin": "https://evil.example"},
        json=body,
    ).status_code == 403

    accepted = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json=body,
    )
    assert accepted.status_code == 202
    submission = worker.submissions[-1]
    assert submission["actor_user_id"] == "alice"
    assert submission["idempotency_key"] == "browser-request-0001"
    assert submission["scope"] == [
        {
            "notebookId": "notebook-a",
            "contentRevision": "sha256:notebook-a",
            "indexSchemaVersion": "novo-chat-index-v1",
        }
    ]
    # The worker adapter accepts only typed job fields; a cookie/raw request has
    # no place in the call boundary.
    assert "cookie" not in submission
    assert "session" not in submission
    assert "request" not in submission


def test_job_status_hides_result_and_release_reauthorizes(gateway) -> None:
    client, novo, _worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    accepted = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={
            "operation": "ask",
            "corpus": "novo:notebook-a",
            "question": "What happened?",
            "model": "approved-model",
        },
    ).json()
    job_id = accepted["jobId"]

    poll = client.get(f"/lab-chat/api/jobs/{job_id}", headers=session_headers())
    assert poll.status_code == 200
    assert "result" not in poll.json()
    released = client.get(f"/lab-chat/api/jobs/{job_id}/result", headers=session_headers())
    assert released.status_code == 200
    assert released.json()["result"]["answer"] == "Grounded answer [1]"

    novo.contexts["alice-session"] = context_for("alice", ())
    revoked = client.get(f"/lab-chat/api/jobs/{job_id}/result", headers=session_headers())
    assert revoked.status_code == 403


def test_job_status_and_result_reject_content_change_with_unchanged_membership(gateway) -> None:
    client, novo, _worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    job_id = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={
            "operation": "ask",
            "corpus": "novo:notebook-a",
            "question": "What happened?",
            "model": "approved-model",
        },
    ).json()["jobId"]

    novo.contexts["alice-session"] = context_for(
        "alice",
        content_revisions={"notebook-a": "sha256:notebook-a-new"},
    )
    for path in (
        f"/lab-chat/api/jobs/{job_id}",
        f"/lab-chat/api/jobs/{job_id}/result",
    ):
        response = client.get(path, headers=session_headers())
        assert response.status_code == 409
        assert response.json() == {
            "detail": "Notebook content changed before job release. Submit a new request."
        }


def test_job_result_rechecks_revision_after_worker_response(gateway) -> None:
    client, novo, worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    job_id = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={
            "operation": "ask",
            "corpus": "novo:notebook-a",
            "question": "What happened?",
            "model": "approved-model",
        },
    ).json()["jobId"]
    original_job_status = worker.job_status

    async def change_revision_during_worker_fetch(requested_job_id: str) -> dict[str, Any]:
        response = await original_job_status(requested_job_id)
        novo.contexts["alice-session"] = context_for(
            "alice",
            content_revisions={"notebook-a": "sha256:notebook-a-new"},
        )
        return response

    worker.job_status = change_revision_during_worker_fetch
    response = client.get(f"/lab-chat/api/jobs/{job_id}/result", headers=session_headers())
    assert response.status_code == 409
    assert response.json() == {
        "detail": "Notebook content changed before job release. Submit a new request."
    }


def test_job_release_rejects_index_schema_change(gateway) -> None:
    client, _novo, _worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    job_id = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={
            "operation": "ask",
            "corpus": "novo:notebook-a",
            "question": "What happened?",
            "model": "approved-model",
        },
    ).json()["jobId"]
    database_path = client.app.state.job_store.path
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT scope_json FROM job_owners WHERE job_id = ?", (job_id,)).fetchone()
        scope = json.loads(row[0])
        scope[0]["indexSchemaVersion"] = "novo-chat-index-v0"
        connection.execute(
            "UPDATE job_owners SET scope_json = ? WHERE job_id = ?",
            (json.dumps(scope), job_id),
        )

    response = client.get(f"/lab-chat/api/jobs/{job_id}/result", headers=session_headers())
    assert response.status_code == 409
    assert response.json()["detail"] == "Notebook content changed before job release. Submit a new request."


def test_job_result_rejects_unsafe_worker_source_url(gateway) -> None:
    client, _novo, worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    accepted = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={
            "operation": "ask",
            "corpus": "novo:notebook-a",
            "question": "What happened?",
            "model": "approved-model",
        },
    ).json()
    job_id = accepted["jobId"]
    worker.jobs[job_id]["result"]["hits"][0]["novoUrl"] = "javascript:alert(document.cookie)"

    response = client.get(f"/lab-chat/api/jobs/{job_id}/result", headers=session_headers())
    assert response.status_code == 502
    assert response.json() == {
        "detail": "Worker result contained an unsafe source URL",
        "error": {"code": "WORKER_REJECTED", "retryable": False},
    }


def test_gateway_uses_small_default_novo_export_batches() -> None:
    assert GatewaySettings().novo_export_page_limit == 8
    assert GatewaySettings().ingest_batch_max_bytes == 24 * 1024 * 1024
    assert GatewaySettings().gateway_max_request_bytes == 64 * 1024
    assert GatewaySettings().job_ownership_ttl_s == 7 * 24 * 60 * 60


def test_queue_full_is_exposed_as_retryable_429(gateway) -> None:
    client, _novo, worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()

    async def reject_queue(**_kwargs: Any) -> dict[str, Any]:
        raise WorkerRejected(
            "Worker queue is full.",
            code="QUEUE_FULL",
            retryable=True,
            status_code=429,
        )

    worker.submit = reject_queue
    response = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={"operation": "model_start", "model": "approved-model"},
    )
    assert response.status_code == 429
    assert response.json() == {
        "detail": "Worker queue is full.",
        "error": {"code": "QUEUE_FULL", "retryable": True},
    }


def test_public_job_poll_preserves_safe_structured_retryability(gateway) -> None:
    client, _novo, worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    job_id = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={"operation": "model_start", "model": "approved-model"},
    ).json()["jobId"]
    worker.jobs[job_id]["state"] = "failed"
    worker.jobs[job_id]["error"] = {
        "code": "MODEL_NOT_READY",
        "message": "Requested model is not ready.",
        "retryable": True,
    }

    response = client.get(f"/lab-chat/api/jobs/{job_id}", headers=session_headers())
    assert response.status_code == 200
    assert response.json()["error"] == {
        "code": "MODEL_NOT_READY",
        "message": "Requested model is not ready.",
        "retryable": True,
    }


def test_model_state_ui_distinguishes_transitions_and_failures() -> None:
    javascript = (Path(__file__).parents[1] / "novo_chat/gateway/web/app.js").read_text(encoding="utf-8")
    assert 'starting: "Starting..."' in javascript
    assert 'draining: "Draining..."' in javascript
    assert 'failed: "Model failed"' in javascript
    assert 'stopping: "Stopping..."' in javascript
    assert 'unavailable: "Model unavailable"' in javascript


def test_gateway_ui_preserves_aorus_layout_and_ranked_retrieval_contract() -> None:
    web_directory = Path(__file__).parents[1] / "novo_chat/gateway/web"
    html = (web_directory / "index.html").read_text(encoding="utf-8")
    css = (web_directory / "styles.css").read_text(encoding="utf-8")
    javascript = (web_directory / "app.js").read_text(encoding="utf-8")

    assert "<span>Corpus</span>" in html
    assert "<span>Context chunks</span>" in html
    assert "<h2>Retrieved</h2>" in html
    assert 'placeholder="Ask a question..."' in html
    assert "Access checked live by Novo" not in html
    assert 'id="accessSource"' not in html
    assert "Back to Novo" in html
    assert "Ask your Novo notebooks" not in html
    assert 'style="' not in html

    assert "width: 288px" in css
    assert "width: 384px" in css
    assert "border-radius: 0" in css

    assert 'addMessage("assistant", "thinking...")' in javascript
    assert 'addMessage("assistant", "Start a model before asking.")' in javascript
    assert "el.userName.textContent = label;" in javascript
    assert "user?.role" not in javascript
    assert "accessSource" not in javascript
    assert "retrieval_top_k: state.retrievalTopK" in javascript
    assert "usedInContext" in javascript
    assert "source-ranked-only" in javascript
    assert "timings?.prompt_eval_count" in javascript
    assert "timings?.num_ctx" in javascript
    assert '["total VRAM", totalVramGb != null && Number.isFinite(Number(totalVramGb))' in javascript
    assert 'style="' not in javascript


def test_gateway_ui_guards_jobs_recovery_and_responsive_sources() -> None:
    web_directory = Path(__file__).parents[1] / "novo_chat/gateway/web"
    css = (web_directory / "styles.css").read_text(encoding="utf-8")
    javascript = (web_directory / "app.js").read_text(encoding="utf-8")

    assert 'const MODEL_MAY_BE_RUNNING = new Set(["ready", "running", "starting", "draining", "failed"]);' in javascript
    assert "const controlsLocked = state.busy || !computeAvailable || state.runtimeAction !== null;" in javascript
    assert "if (shouldRefreshIndex) await refreshIndexStatus();" in javascript
    assert "if (worker.available !== true) {" in javascript
    assert "el.corpus.disabled = selectionLocked" in javascript
    assert "el.model.disabled = selectionLocked" in javascript
    assert "el.maxSources.disabled = selectionLocked" in javascript
    assert "if (!selectionMatches(selection))" in javascript
    assert "state.rebuildingCorpus === state.corpus" in javascript
    assert "state.runtimeModel === state.model" in javascript

    responsive = css[css.index("@media (max-width: 1080px)") :]
    assert "display: none" not in responsive
    assert "grid-template-rows: minmax(0, 3fr) minmax(220px, 2fr)" in responsive
    assert "grid-row: 3" in responsive


def test_gateway_accepts_legacy_max_sources_limit_but_bounds_retrieval_depth() -> None:
    common = {
        "operation": "ask",
        "corpus": "novo:notebook-a",
        "question": "What happened?",
        "model": "approved-model",
    }
    accepted = GatewayJobRequest.model_validate({**common, "max_sources": 100, "retrieval_top_k": 32})
    assert accepted.max_sources == 100
    assert accepted.retrieval_top_k == 32

    with pytest.raises(ValueError):
        GatewayJobRequest.model_validate({**common, "max_sources": 101})
    with pytest.raises(ValueError):
        GatewayJobRequest.model_validate({**common, "retrieval_top_k": 33})


def test_job_ownership_is_not_disclosed_to_another_user(gateway) -> None:
    client, _novo, _worker = gateway
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    job_id = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={"operation": "index_rebuild", "corpus": "novo:notebook-a", "force": True},
    ).json()["jobId"]

    response = client.get(f"/lab-chat/api/jobs/{job_id}", headers=session_headers("bob-session"))
    assert response.status_code == 404


def test_compute_unavailable_has_a_stable_api_state(gateway) -> None:
    client, _novo, worker = gateway
    worker.available = False
    context = client.get("/lab-chat/api/context", headers=session_headers()).json()
    assert context["worker"]["available"] is False
    response = client.post(
        "/lab-chat/api/jobs",
        headers=auth_headers(context["csrfToken"]),
        json={"operation": "model_start", "model": "approved-model"},
    )
    assert response.status_code == 503
    assert response.json()["detail"].startswith("The compute service is unavailable")
