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
                        "updatedAt": "2026-07-31T20:00:00Z",
                        "pageCount": 42,
                        "attachmentCount": 7,
                        "textChars": 18240,
                    },
                    {
                        "id": "notebook-beta",
                        "name": "Cell Assays",
                        "accessRole": "viewer",
                        "contentRevision": "sha256:" + "b" * 64,
                        "updatedAt": "2026-07-31T19:30:00Z",
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
        self.model_ready = False
        self.query_count = 0

    async def health(self):
        return {"state": "ready", "queueHealthy": True, "indexServiceHealthy": True}

    async def capabilities(self):
        return {
            "models": ["demo:model"],
            "operations": ["query", "index_rebuild", "model_start", "model_stop"],
            "modelDetails": {"demo:model": {"modelSize": "27B", "maxTokens": 2048, "maxModelLen": 32768, "thinking": "enabled", "totalVramGb": 96}},
        }

    async def model_status(self):
        state = "ready" if self.model_ready else "stopped"
        return {"models": {"demo:model": {"state": state, "healthy": self.model_ready}}}

    async def index_status(self, *, request_id, actor_user_id, scope):
        del request_id, actor_user_id
        rows = []
        for entry in scope:
            exact_ready = entry["notebookId"] == "notebook-alpha"
            row = {**entry, "exactReady": exact_ready}
            if exact_ready:
                row.update({"activatedAt": "2026-07-31T20:15:00Z", "chunkCount": 92})
            rows.append(row)
        return {"indexes": rows}

    async def submit(self, *, operation, request_id, idempotency_key, actor_user_id, scope, payload):
        del request_id, idempotency_key, actor_user_id
        job_id = f"job_{uuid4().hex}"
        if operation == "query":
            self.query_count += 1
            followup = self.query_count > 1
            page_id = "page-controls" if followup else "page-result"
            title = "Control measurements" if followup else "Dose response experiment"
            excerpt = "Untreated controls were measured at baseline." if followup else "The treated samples showed an increased response."
            notebook_id = scope[0]["notebookId"]
            plan = {
                "originalQuestion": payload["question"],
                "semanticQuery": "treatment effect recorded response experiment",
                "bm25Terms": ["treatment", "response", "experiment"],
                "mode": "planned",
            }
            result = {
                "kind": "query",
                "answer": "Untreated controls were measured at baseline [1]." if followup else "The recorded response increased after treatment [1]. The repeat measurement agreed [2].",
                "model": payload["model"],
                "retrievalPlan": plan,
                "timings": {"prompt_eval_count": 4096 if followup else 8192, "eval_count": 2048 if followup else 4096, "num_ctx": 32768},
                "citations": [
                    {
                        "notebookId": notebook_id,
                        "pageId": page_id,
                        "sourceUrl": f"/?page={page_id}",
                        "title": title,
                        "file": f"{page_id}.md",
                        "excerpt": excerpt,
                        "sourceIdx": 1,
                        "usedInContext": True,
                        "score": 0.91,
                    },
                    {"notebookId": notebook_id, "pageId": page_id, "title": title,
                     "excerpt": "A second measurement confirmed the result.", "sourceIdx": 2,
                     "sourceUrl": f"/?page={page_id}", "usedInContext": True, "score": 0.88},
                    {"notebookId": notebook_id, "pageId": f"{page_id}-protocol", "title": f"{title}: protocol",
                     "excerpt": "The protocol records sample preparation and instrument settings.", "sourceIdx": 3,
                     "sourceUrl": f"/?page={page_id}-protocol", "usedInContext": False, "score": 0.81},
                ],
            }
        elif operation == "index_rebuild":
            result = {"kind": "index", "indexes": scope, "chunkCount": 128}
        elif operation == "model_start":
            result = {"kind": "model", "model": payload["model"], "state": "ready"}
        else:
            self.model_ready = False
            result = {"kind": "model", "model": payload["model"], "state": "stopped"}
        job = {
            "jobId": job_id,
            "operation": operation,
            "state": "queued" if operation in {"query", "model_start"} else "succeeded",
            "progress": 0.0 if operation in {"query", "model_start"} else 1.0,
            "result": result,
            "_polls": 0,
        }
        self.jobs[job_id] = job
        return {"job": job}

    async def job_status(self, job_id):
        job = self.jobs[job_id]
        if job["operation"] == "query":
            job["_polls"] += 1
            plan = job["result"]["retrievalPlan"]
            if job["_polls"] == 1:
                job.update(
                    {
                        "state": "running",
                        "progress": 0.05,
                        "progressDetail": {"stage": "planning"},
                    }
                )
            elif job["_polls"] <= 4:
                job.update(
                    {
                        "progress": 0.30,
                        "progressDetail": {
                            "stage": "searching",
                            "retrievalPlan": plan,
                        },
                    }
                )
            elif job["_polls"] <= 12:
                job.update(
                    {
                        "progress": 0.65,
                        "progressDetail": {
                            "stage": "answering",
                            "retrievalPlan": plan,
                            "retrievedCount": 2,
                        },
                    }
                )
            else:
                job.update({"state": "succeeded", "progress": 1.0})
        elif job["operation"] == "model_start":
            job["_polls"] += 1
            if job["_polls"] == 1:
                job.update({"state": "running", "progress": 0.03})
            elif job["_polls"] == 2:
                job.update({"progress": 0.49})
            elif job["_polls"] == 3:
                job.update({"progress": 1.0})
            else:
                self.model_ready = True
                job.update({"state": "succeeded", "progress": 1.0})
        return {"job": {key: value for key, value in job.items() if not key.startswith("_")}}

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
