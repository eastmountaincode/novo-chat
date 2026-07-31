from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from novo_chat.gateway.novo_client import NovoIntegrationClient, NovoRevisionChanged
from novo_chat.gateway.secrets import FileSecret, write_test_secret
from novo_chat.gateway.worker_client import WorkerRejected
from novo_chat.gateway.worker_http_client import WorkerHttpClient
from novo_chat.protocol import (
    ErrorResponse,
    JobError,
    JobOperation,
    JobState,
    JobSubmissionResponse,
    JobView,
    QueryJobResult,
    canonical_json,
    ingest_pages_checksum,
    sign_response,
    verify_request,
)


def test_novo_client_forwards_only_named_session_to_loopback(tmp_path: Path) -> None:
    secret_path = tmp_path / "novo-secret"
    write_test_secret(secret_path, "novo-integration-secret-0123456789")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://127.0.0.1:3148/api/integrations/v1/context")
        assert request.headers["authorization"] == "Bearer novo-integration-secret-0123456789"
        assert request.headers["cookie"] == "eln_session=opaque-browser-session"
        return httpx.Response(
            200,
            json={
                "apiVersion": "1",
                "novoVersion": "test",
                "user": {"id": "user-1", "email": "user@example.org"},
                "notebooks": [],
            },
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = NovoIntegrationClient(
                base_url="http://127.0.0.1:3148/api/integrations/v1",
                secret=FileSecret(secret_path),
                session_cookie_name="eln_session",
                client=http,
            )
            context = await client.context("opaque-browser-session")
            assert context.user.id == "user-1"

    asyncio.run(exercise())


def test_novo_client_exports_typed_pages_with_exact_revision_and_cursor(tmp_path: Path) -> None:
    secret_path = tmp_path / "novo-secret"
    write_test_secret(secret_path, "novo-integration-secret-0123456789")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"].startswith("Bearer ")
        assert request.headers["cookie"] == "eln_session=opaque-browser-session"
        assert request.headers["if-match"] == '"sha256:revision-a"'
        cursor = request.url.params.get("cursor")
        pages = [
            {
                "id": "page-a" if cursor is None else "page-b",
                "title": "Page",
                "text": "Normalized content",
                "createdAt": "2026-07-30T12:00:00Z",
                "updatedAt": "2026-07-31T12:00:00Z",
                "tags": ["tag"],
                "attachments": [],
                "sourceUrl": "/?page=page-a",
            }
        ]
        return httpx.Response(
            200,
            json={
                "apiVersion": "1",
                "notebook": {
                    "id": "notebook-a",
                    "name": "Notebook A",
                    "contentRevision": "sha256:revision-a",
                },
                "pages": pages,
                "nextCursor": "cursor-2" if cursor is None else None,
                "complete": cursor is not None,
            },
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = NovoIntegrationClient(
                base_url="http://127.0.0.1:3148/api/integrations/v1",
                secret=FileSecret(secret_path),
                session_cookie_name="eln_session",
                client=http,
            )
            first = await client.export_pages(
                "opaque-browser-session",
                notebook_id="notebook-a",
                content_revision="sha256:revision-a",
                limit=100,
            )
            second = await client.export_pages(
                "opaque-browser-session",
                notebook_id="notebook-a",
                content_revision="sha256:revision-a",
                cursor=first.next_cursor,
                limit=100,
            )
            assert first.complete is False
            assert second.complete is True
            worker_document = second.pages[0].worker_document()
            assert worker_document["pageId"] == "page-b"
            assert worker_document["createdAt"] == "2026-07-30T12:00:00Z"
            assert "updatedAt" not in worker_document

    asyncio.run(exercise())
    assert dict(requests[0].url.params) == {"limit": "100"}
    assert dict(requests[1].url.params) == {"limit": "100", "cursor": "cursor-2"}


def test_novo_client_surfaces_precondition_failure_as_revision_change(tmp_path: Path) -> None:
    secret_path = tmp_path / "novo-secret"
    write_test_secret(secret_path, "novo-integration-secret-0123456789")

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(412))) as http:
            client = NovoIntegrationClient(
                base_url="http://127.0.0.1:3148/api/integrations/v1",
                secret=FileSecret(secret_path),
                session_cookie_name="eln_session",
                client=http,
            )
            try:
                await client.export_pages(
                    "opaque-browser-session",
                    notebook_id="notebook-a",
                    content_revision="sha256:revision-a",
                )
            except NovoRevisionChanged as exc:
                assert exc.notebook_id == "notebook-a"
            else:
                raise AssertionError("expected a distinct revision-change signal")

    asyncio.run(exercise())


def test_worker_client_signs_exact_typed_body_and_has_no_cookie_channel(tmp_path: Path) -> None:
    request_secret_path = tmp_path / "gateway-key"
    response_secret_path = tmp_path / "worker-key"
    request_secret = "gateway-request-secret-01234567890123456789"
    response_secret = "worker-response-secret-01234567890123456789"
    write_test_secret(request_secret_path, request_secret)
    write_test_secret(response_secret_path, response_secret)
    seen_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        verified = verify_request(
            headers=request.headers,
            secrets_by_key_id={"gateway-key": request_secret},
            expected_environment="staging",
            method=request.method,
            path=request.url.path,
            body=body,
        )
        assert request.url.path == "/internal/v1/query"
        assert "cookie" not in request.headers
        seen_body.update(json.loads(body))
        response_model = JobSubmissionResponse(
            request_id=verified.request_id,
            job=JobView(
                job_id="job-1",
                operation=JobOperation.QUERY,
                state=JobState.SUCCEEDED,
                created_at="2026-07-31T12:00:00Z",
                updated_at="2026-07-31T12:00:01Z",
                progress=1.0,
                result=QueryJobResult(answer="Answer", model="approved-model", citations=()),
            ),
        )
        response_body = canonical_json(response_model)
        headers = sign_response(
            secret=response_secret,
            key_id="worker-key",
            environment="staging",
            request_id=verified.request_id,
            status_code=202,
            body=response_body,
        )
        return httpx.Response(202, headers=headers, content=response_body)

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = WorkerHttpClient(
                base_url="http://127.0.0.1:8196/internal/v1",
                request_secret=FileSecret(request_secret_path, minimum_bytes=32),
                response_secret=FileSecret(response_secret_path, minimum_bytes=32),
                request_key_id="gateway-key",
                response_key_id="worker-key",
                environment="staging",
                timeout_s=2,
                client=http,
            )
            result = await client.submit(
                operation="query",
                request_id="ce0bcb9c-8cbf-4d1c-8ca9-8eeec189e91c",
                idempotency_key="idempotency-key-1",
                actor_user_id="user-1",
                scope=[
                    {
                        "notebookId": "notebook-1",
                        "contentRevision": "sha256:abc123",
                        "indexSchemaVersion": "novo-chat-v1",
                    }
                ],
                payload={
                    "question": "What happened?",
                    "retrieval_question": "What happened?",
                    "model": "approved-model",
                    "strategy": "hybrid",
                    "max_sources": 6,
                },
            )
            assert result["job"]["jobId"] == "job-1"

    asyncio.run(exercise())
    assert seen_body["actorUserId"] == "user-1"
    assert seen_body["scope"] == [
        {
            "notebookId": "notebook-1",
            "contentRevision": "sha256:abc123",
            "indexSchemaVersion": "novo-chat-v1",
        }
    ]


def test_worker_client_computes_canonical_ingest_checksum(tmp_path: Path) -> None:
    request_secret_path = tmp_path / "gateway-key"
    response_secret_path = tmp_path / "worker-key"
    request_secret = "gateway-request-secret-01234567890123456789"
    response_secret = "worker-response-secret-01234567890123456789"
    write_test_secret(request_secret_path, request_secret)
    write_test_secret(response_secret_path, response_secret)
    page = {
        "pageId": "page-1",
        "title": "Title",
        "text": "Text",
        "sourceUrl": "/?page=page-1",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        verified = verify_request(
            headers=request.headers,
            secrets_by_key_id={"gateway-key": request_secret},
            expected_environment="staging",
            method=request.method,
            path=request.url.path,
            body=request.content,
        )
        body = json.loads(request.content)
        assert body["batchChecksum"] == ingest_pages_checksum([page])
        assert body["pages"][0]["attachments"] == []
        assert "cookie" not in request.headers
        response_model = JobSubmissionResponse(
            request_id=verified.request_id,
            job=JobView(
                job_id="job-ingest",
                operation=JobOperation.INGEST_BATCH,
                state=JobState.SUCCEEDED,
                created_at="2026-07-31T12:00:00Z",
                updated_at="2026-07-31T12:00:01Z",
                progress=1.0,
            ),
        )
        response_body = canonical_json(response_model)
        return httpx.Response(
            202,
            headers=sign_response(
                secret=response_secret,
                key_id="worker-key",
                environment="staging",
                request_id=verified.request_id,
                status_code=202,
                body=response_body,
            ),
            content=response_body,
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = WorkerHttpClient(
                base_url="http://127.0.0.1:8196/internal/v1",
                request_secret=FileSecret(request_secret_path, minimum_bytes=32),
                response_secret=FileSecret(response_secret_path, minimum_bytes=32),
                request_key_id="gateway-key",
                response_key_id="worker-key",
                environment="staging",
                timeout_s=2,
                client=http,
            )
            response = await client.ingest_batch(
                request_id="ce0bcb9c-8cbf-4d1c-8ca9-8eeec189e91c",
                idempotency_key="ingest-idempotency-1",
                actor_user_id="user-1",
                scope={
                    "notebookId": "notebook-1",
                    "contentRevision": "sha256:abc123",
                    "indexSchemaVersion": "novo-chat-index-v1",
                },
                batch_number=0,
                pages=[page],
            )
            assert response["job"]["operation"] == "ingest_batch"

    asyncio.run(exercise())


def test_worker_client_preserves_signed_queue_full_retryability(tmp_path: Path) -> None:
    request_secret_path = tmp_path / "gateway-key"
    response_secret_path = tmp_path / "worker-key"
    request_secret = "gateway-request-secret-01234567890123456789"
    response_secret = "worker-response-secret-01234567890123456789"
    write_test_secret(request_secret_path, request_secret)
    write_test_secret(response_secret_path, response_secret)

    def handler(request: httpx.Request) -> httpx.Response:
        verified = verify_request(
            headers=request.headers,
            secrets_by_key_id={"gateway-key": request_secret},
            expected_environment="staging",
            method=request.method,
            path=request.url.path,
            body=request.content,
        )
        response_model = ErrorResponse(
            request_id=verified.request_id,
            error=JobError(code="QUEUE_FULL", message="Worker queue is full.", retryable=True),
        )
        response_body = canonical_json(response_model)
        return httpx.Response(
            429,
            headers=sign_response(
                secret=response_secret,
                key_id="worker-key",
                environment="staging",
                request_id=verified.request_id,
                status_code=429,
                body=response_body,
            ),
            content=response_body,
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = WorkerHttpClient(
                base_url="http://127.0.0.1:8196/internal/v1",
                request_secret=FileSecret(request_secret_path, minimum_bytes=32),
                response_secret=FileSecret(response_secret_path, minimum_bytes=32),
                request_key_id="gateway-key",
                response_key_id="worker-key",
                environment="staging",
                timeout_s=2,
                client=http,
            )
            try:
                await client.submit(
                    operation="model_start",
                    request_id="ce0bcb9c-8cbf-4d1c-8ca9-8eeec189e91c",
                    idempotency_key="model-start-request-1",
                    actor_user_id="user-1",
                    scope=[],
                    payload={"model": "approved-model"},
                )
            except WorkerRejected as exc:
                assert exc.code == "QUEUE_FULL"
                assert exc.retryable is True
                assert exc.status_code == 429
                assert str(exc) == "Worker queue is full."
            else:
                raise AssertionError("expected queue-full rejection")

    asyncio.run(exercise())
