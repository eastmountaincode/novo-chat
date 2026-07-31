"""Local rendered-browser fixture; never used by deployment packaging."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi.responses import RedirectResponse

from novo_chat.gateway.app import create_app
from novo_chat.gateway.config import GatewaySettings
from novo_chat.gateway.novo_client import NovoContext
from novo_chat.gateway.secrets import write_test_secret


_temporary = tempfile.TemporaryDirectory(prefix="novo-chat-browser-")
_root = Path(_temporary.name)
for _name, _value in (
    ("novo", "n" * 48),
    ("csrf", "c" * 48),
    ("request", "q" * 48),
    ("response", "r" * 48),
):
    write_test_secret(_root / _name, _value)


class FixtureNovo:
    async def context(self, _session_value: str) -> NovoContext:
        return NovoContext.model_validate(
            {
                "apiVersion": "1",
                "novoVersion": "browser-fixture",
                "user": {
                    "id": "fixture-user",
                    "email": "researcher@example.test",
                    "firstName": "Test",
                    "lastName": "Researcher",
                },
                "notebooks": [
                    {
                        "id": "notebook-alpha",
                        "name": "Protein Engineering",
                        "accessRole": "owner",
                        "contentRevision": "sha256:" + "a" * 64,
                        "pageCount": 42,
                        "attachmentCount": 7,
                        "textChars": 18240,
                    },
                    {
                        "id": "notebook-beta",
                        "name": "Cell Assays",
                        "accessRole": "viewer",
                        "contentRevision": "sha256:" + "b" * 64,
                        "pageCount": 18,
                        "attachmentCount": 2,
                        "textChars": 7310,
                    },
                ],
            }
        )


class FixtureWorker:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    async def health(self):
        return {"state": "ready", "queueHealthy": True, "indexServiceHealthy": True}

    async def capabilities(self):
        return {"models": ["demo:model"], "operations": ["query", "index_rebuild"]}

    async def model_status(self):
        return {"models": {"demo:model": {"state": "ready", "healthy": True}}}

    async def index_status(self, *, request_id, actor_user_id, scope):
        del request_id, actor_user_id
        return {"indexes": [{**entry, "exactReady": True} for entry in scope]}

    async def submit(self, *, operation, request_id, idempotency_key, actor_user_id, scope, payload):
        del request_id, idempotency_key, actor_user_id
        job_id = f"job_{uuid4().hex}"
        if operation == "query":
            notebook_id = scope[0]["notebookId"]
            result = {
                "kind": "query",
                "answer": "The recorded response increased after treatment [1].",
                "model": payload["model"],
                "citations": [
                    {
                        "notebookId": notebook_id,
                        "pageId": "page-result",
                        "sourceUrl": "/?page=page-result",
                        "title": "Dose response experiment",
                        "excerpt": "The treated samples showed an increased response.",
                        "score": 0.91,
                    }
                ],
            }
        elif operation == "index_rebuild":
            result = {"kind": "index", "indexes": scope, "chunkCount": 128}
        else:
            result = {"kind": "model", "model": payload["model"], "state": "ready"}
        job = {
            "jobId": job_id,
            "operation": operation,
            "state": "succeeded",
            "progress": 1.0,
            "result": result,
        }
        self.jobs[job_id] = job
        return {"job": job}

    async def job_status(self, job_id):
        return {"job": self.jobs[job_id]}

    async def aclose(self):
        return None


_settings = GatewaySettings(
    base_path="/chat",
    novo_api_base_url="http://127.0.0.1:3148/api/integrations/v1",
    novo_integration_secret_file=_root / "novo",
    csrf_secret_file=_root / "csrf",
    worker_base_url="http://127.0.0.1:8195/internal/v1",
    worker_signing_secret_file=_root / "request",
    worker_response_secret_file=_root / "response",
    environment="staging",
    public_origin="http://127.0.0.1:8765",
    job_db_path=_root / "jobs.sqlite3",
)

app = create_app(_settings, novo_client=FixtureNovo(), worker_client=FixtureWorker())


@app.get("/test-login", include_in_schema=False)
async def fixture_login() -> RedirectResponse:
    response = RedirectResponse("/chat/", status_code=303)
    response.set_cookie("eln_session", "fixture-session", httponly=True, samesite="lax")
    return response
