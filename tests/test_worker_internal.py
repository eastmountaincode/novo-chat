from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from novo_chat.documents import FinalizedDocument
from novo_chat.jobs import JobNotFound, JobStore
from novo_chat.protocol import (
    JobOperation,
    ModelDisplayDetails,
    NotebookScope,
    canonical_json,
    document_pages_checksum,
    ingest_pages_checksum,
    sign_request,
    verify_response,
)
from novo_chat.worker import WorkerConfig, create_worker_app


REQUEST_SECRET = b"request-secret-0123456789abcdef0"
RESPONSE_SECRET = b"response-secret-0123456789abcdef"
NOW = 1_700_000_000


async def fail_if_request_stream_is_read(_request):
    raise AssertionError("rejected request body was read")
    yield b""


class WorkerInternalAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.config = WorkerConfig(
            environment="staging",
            state_database_path=f"{self.temporary_directory.name}/worker.sqlite3",
            request_secrets_by_key_id={"gateway": REQUEST_SECRET},
            response_key_id="worker",
            response_secret=RESPONSE_SECRET,
            approved_models=("model:a",),
            model_details={
                "model:a": ModelDisplayDetails(
                    modelSize="7B",
                    maxTokens=2048,
                    maxModelLen=32768,
                    thinking="disabled",
                    totalVramGb=24,
                )
            },
            index_schema_versions=("index-v1",),
        )
        self.app = create_worker_app(self.config, clock=lambda: NOW, auto_start_executor=False)
        self.client = TestClient(self.app)
        self.scope = {
            "notebookId": "notebook-a",
            "contentRevision": "sha256:revision-a",
            "indexSchemaVersion": "index-v1",
        }

    def signed_request(self, method: str, path: str, payload=None, *, request_id=None, nonce=None):
        request_id = request_id or (payload or {}).get("requestId") or str(uuid4())
        body = b"" if payload is None else canonical_json(payload)
        headers = sign_request(
            secret=REQUEST_SECRET,
            key_id="gateway",
            environment="staging",
            request_id=request_id,
            method=method,
            path=path,
            body=body,
            timestamp=NOW,
            nonce=nonce,
        )
        if payload is not None:
            headers["Content-Type"] = "application/json"
        response = self.client.request(method, path, content=body, headers=headers)
        verify_response(
            headers=response.headers,
            secrets_by_key_id={"worker": RESPONSE_SECRET},
            expected_environment="staging",
            request_id=request_id,
            status_code=response.status_code,
            body=response.content,
            now=NOW,
        )
        return response, headers, body

    def job_payload(self, **extra):
        payload = {
            "requestId": str(uuid4()),
            "idempotencyKey": f"idem-{uuid4()}",
            "actorUserId": "user-a",
            "scope": [self.scope],
        }
        payload.update(extra)
        return payload

    def test_health_requires_auth_and_authenticated_response_is_signed(self):
        unsigned = self.client.get("/internal/v1/health")
        self.assertEqual(unsigned.status_code, 401)
        self.assertEqual(unsigned.json()["error"]["code"], "AUTHENTICATION_REQUIRED")
        self.assertIn("X-Novo-Chat-Signature", unsigned.headers)
        response, _, _ = self.signed_request("GET", "/internal/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "ready")
        self.assertEqual(response.json()["environment"], "staging")

    def test_capabilities_publish_typed_model_display_metadata(self):
        response, _, _ = self.signed_request("GET", "/internal/v1/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["modelDetails"],
            {
                "model:a": {
                    "modelSize": "7B",
                    "maxTokens": 2048,
                    "maxModelLen": 32768,
                    "thinking": "disabled",
                    "totalVramGb": 24.0,
                }
            },
        )

    def test_index_status_reports_metadata_only_for_a_validated_active_artifact(self):
        scope = NotebookScope.model_validate(self.scope)
        candidate = self.app.state.index_repository.build_candidate(
            FinalizedDocument(
                scope=scope,
                document_checksum=document_pages_checksum([]),
                pages=(),
            ),
            operation_id="op-status-metadata",
        )
        committed = self.app.state.index_repository.commit_candidate(candidate)
        self.app.state.job_store.activate_index(
            scope,
            environment="staging",
            artifact_id=committed.artifact_id,
            now=NOW,
        )

        payload = {
            "requestId": str(uuid4()),
            "actorUserId": "user-a",
            "scope": [self.scope],
        }
        ready, _, _ = self.signed_request("POST", "/internal/v1/indexes/status", payload)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(
            ready.json()["indexes"][0],
            {
                **self.scope,
                "exactReady": True,
                "activatedAt": "2023-11-14T22:13:20Z",
                "chunkCount": 0,
            },
        )

        manifest_path = (
            self.app.state.index_repository.artifact_root
            / committed.artifact_id
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["chunkCount"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        payload["requestId"] = str(uuid4())
        corrupt, _, _ = self.signed_request("POST", "/internal/v1/indexes/status", payload)
        self.assertEqual(corrupt.status_code, 200)
        self.assertEqual(corrupt.json()["indexes"][0], {**self.scope, "exactReady": False})

    def test_job_retention_configuration_is_positive_integer_seconds(self):
        self.assertEqual(self.config.job_retention_seconds, 7 * 24 * 60 * 60)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config, job_retention_seconds=value)
        for field in (
            "orphan_grace_seconds",
            "document_storage_quota_bytes",
            "index_storage_quota_bytes",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.config, **{field: 0})
        with self.assertRaises(ValueError):
            replace(
                self.config,
                job_retention_seconds=60,
                orphan_grace_seconds=61,
            )

    def test_startup_purges_expired_jobs_and_active_index_pointers(self):
        database_path = f"{self.temporary_directory.name}/startup-retention.sqlite3"
        state_directory = f"{self.temporary_directory.name}/startup-retention-state"
        store = JobStore(database_path)
        scope = NotebookScope.model_validate(self.scope)
        old = store.submit(
            environment="staging",
            actor_user_id="user-old",
            idempotency_key="old-startup-query",
            request_id=str(uuid4()),
            operation=JobOperation.QUERY,
            scope=(scope,),
            payload={"question": "old question", "model": "model:a"},
            now=NOW - 1_000,
        )
        store.succeed(
            old.job.operation_id,
            {"kind": "query", "answer": "old answer", "model": "model:a", "citations": []},
            now=NOW - 900,
        )
        store.activate_index(
            scope,
            environment="staging",
            artifact_id="active-artifact",
            now=NOW - 1_000,
        )

        app = create_worker_app(
            replace(
                self.config,
                state_database_path=database_path,
                state_directory=state_directory,
                job_retention_seconds=100,
                orphan_grace_seconds=50,
            ),
            clock=lambda: NOW,
            auto_start_executor=False,
        )

        with self.assertRaises(JobNotFound):
            app.state.job_store.get(old.job.job_id, environment="staging")
        active = app.state.job_store.active_index(scope, environment="staging")
        self.assertIsNone(active)

    def test_job_submission_opportunistically_purges_expired_terminal_jobs(self):
        database_path = f"{self.temporary_directory.name}/submission-retention.sqlite3"
        app = create_worker_app(
            replace(
                self.config,
                state_database_path=database_path,
                state_directory=f"{self.temporary_directory.name}/submission-retention-state",
                job_retention_seconds=100,
                orphan_grace_seconds=50,
            ),
            clock=lambda: NOW,
            auto_start_executor=False,
        )
        old = app.state.job_store.submit(
            environment="staging",
            actor_user_id="user-old",
            idempotency_key="old-opportunistic-query",
            request_id=str(uuid4()),
            operation=JobOperation.QUERY,
            scope=(NotebookScope.model_validate(self.scope),),
            payload={"question": "old question", "model": "model:a"},
            now=NOW - 1_000,
        )
        app.state.job_store.succeed(
            old.job.operation_id,
            {"kind": "query", "answer": "old answer", "model": "model:a", "citations": []},
            now=NOW - 900,
        )

        old_app, old_client = self.app, self.client
        self.app, self.client = app, TestClient(app)
        try:
            payload = self.job_payload(model="model:a", scope=[])
            accepted, _, _ = self.signed_request("POST", "/internal/v1/models/start", payload)
        finally:
            self.app, self.client = old_app, old_client

        self.assertEqual(accepted.status_code, 202)
        with self.assertRaises(JobNotFound):
            app.state.job_store.get(old.job.job_id, environment="staging")

    def test_health_degrades_if_automatic_executor_stops(self):
        app = create_worker_app(self.config, clock=lambda: NOW, auto_start_executor=True)
        with TestClient(app) as client:
            old_app, old_client = self.app, self.client
            self.app, self.client = app, client
            try:
                healthy, _, _ = self.signed_request("GET", "/internal/v1/health")
                self.assertEqual(healthy.json()["state"], "ready")
                app.state.executor.stop()
                degraded, _, _ = self.signed_request("GET", "/internal/v1/health")
                self.assertEqual(degraded.json()["state"], "degraded")
                self.assertFalse(degraded.json()["queueHealthy"])
            finally:
                self.app, self.client = old_app, old_client

    def test_replayed_signed_request_is_rejected(self):
        request_id = str(uuid4())
        path = "/internal/v1/capabilities"
        response, headers, body = self.signed_request(
            "GET", path, request_id=request_id, nonce="same-nonce-123456789"
        )
        self.assertEqual(response.status_code, 200)
        replay = self.client.request("GET", path, content=body, headers=headers)
        self.assertEqual(replay.status_code, 409)
        self.assertEqual(replay.json()["error"]["code"], "REQUEST_REPLAYED")
        verify_response(
            headers=replay.headers,
            secrets_by_key_id={"worker": RESPONSE_SECRET},
            expected_environment="staging",
            request_id=request_id,
            status_code=409,
            body=replay.content,
            now=NOW,
        )

    def test_declared_oversize_request_is_rejected_before_reading_and_response_is_signed(self):
        request_id = str(uuid4())
        path = "/internal/v1/query"
        headers = sign_request(
            secret=REQUEST_SECRET,
            key_id="gateway",
            environment="staging",
            request_id=request_id,
            method="POST",
            path=path,
            body=b"",
            timestamp=NOW,
        )
        headers["Content-Length"] = str(self.config.max_request_bytes + 1)
        with patch(
            "starlette.requests.Request.stream",
            new=fail_if_request_stream_is_read,
        ):
            response = self.client.post(path, content=b"", headers=headers)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "REQUEST_TOO_LARGE")
        verify_response(
            headers=response.headers,
            secrets_by_key_id={"worker": RESPONSE_SECRET},
            expected_environment="staging",
            request_id=request_id,
            status_code=413,
            body=response.content,
            now=NOW,
        )

    def test_malformed_content_length_is_rejected_before_reading_and_response_is_signed(self):
        path = "/internal/v1/query"
        for content_length in ("-1", "+1", "1.0", "1, 1", "not-a-number"):
            with self.subTest(content_length=content_length):
                request_id = str(uuid4())
                headers = sign_request(
                    secret=REQUEST_SECRET,
                    key_id="gateway",
                    environment="staging",
                    request_id=request_id,
                    method="POST",
                    path=path,
                    body=b"",
                    timestamp=NOW,
                )
                headers["Content-Length"] = content_length
                with patch(
                    "starlette.requests.Request.stream",
                    new=fail_if_request_stream_is_read,
                ):
                    response = self.client.post(path, content=b"", headers=headers)

                self.assertEqual(response.status_code, 400)
                expected_code = (
                    "AMBIGUOUS_REQUEST_FRAMING" if "," in content_length else "INVALID_CONTENT_LENGTH"
                )
                self.assertEqual(response.json()["error"]["code"], expected_code)
                verify_response(
                    headers=response.headers,
                    secrets_by_key_id={"worker": RESPONSE_SECRET},
                    expected_environment="staging",
                    request_id=request_id,
                    status_code=400,
                    body=response.content,
                    now=NOW,
                )

    def test_post_without_content_length_is_rejected_before_hmac(self):
        request_id = str(uuid4())
        path = "/internal/v1/query"
        headers = sign_request(
            secret=REQUEST_SECRET,
            key_id="gateway",
            environment="staging",
            request_id=request_id,
            method="POST",
            path=path,
            body=b"",
            timestamp=NOW,
        )
        request = self.client.build_request("POST", path, content=b"", headers=headers)
        del request.headers["content-length"]
        with patch("novo_chat.worker.verify_request") as verify:
            response = self.client.send(request)
        verify.assert_not_called()
        self.assertEqual(response.status_code, 411)
        self.assertEqual(response.json()["error"]["code"], "LENGTH_REQUIRED")
        verify_response(
            headers=response.headers,
            secrets_by_key_id={"worker": RESPONSE_SECRET},
            expected_environment="staging",
            request_id=request_id,
            status_code=411,
            body=response.content,
            now=NOW,
        )

    def test_chunked_request_is_rejected_before_reading_or_hmac(self):
        request_id = str(uuid4())
        path = "/internal/v1/query"
        headers = sign_request(
            secret=REQUEST_SECRET,
            key_id="gateway",
            environment="staging",
            request_id=request_id,
            method="POST",
            path=path,
            body=b"{}",
            timestamp=NOW,
        )
        headers["Transfer-Encoding"] = "chunked"
        with patch(
            "starlette.requests.Request.stream",
            new=fail_if_request_stream_is_read,
        ), patch("novo_chat.worker.verify_request") as verify:
            response = self.client.post(path, content=b"{}", headers=headers)
        verify.assert_not_called()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "UNSUPPORTED_TRANSFER_ENCODING")
        verify_response(
            headers=response.headers,
            secrets_by_key_id={"worker": RESPONSE_SECRET},
            expected_environment="staging",
            request_id=request_id,
            status_code=400,
            body=response.content,
            now=NOW,
        )

    def test_duplicate_content_lengths_are_rejected_before_reading_or_hmac(self):
        request_id = str(uuid4())
        path = "/internal/v1/query"
        signed = sign_request(
            secret=REQUEST_SECRET,
            key_id="gateway",
            environment="staging",
            request_id=request_id,
            method="POST",
            path=path,
            body=b"",
            timestamp=NOW,
        )
        headers = [*signed.items(), ("Content-Length", "0"), ("Content-Length", "0")]
        with patch(
            "starlette.requests.Request.stream",
            new=fail_if_request_stream_is_read,
        ), patch("novo_chat.worker.verify_request") as verify:
            response = self.client.post(path, content=b"", headers=headers)
        verify.assert_not_called()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "AMBIGUOUS_REQUEST_FRAMING")
        verify_response(
            headers=response.headers,
            secrets_by_key_id={"worker": RESPONSE_SECRET},
            expected_environment="staging",
            request_id=request_id,
            status_code=400,
            body=response.content,
            now=NOW,
        )

    def test_actual_body_size_is_checked_when_content_length_is_understated(self):
        config = replace(self.config, max_request_bytes=1024)
        app = create_worker_app(config, clock=lambda: NOW, auto_start_executor=False)
        client = TestClient(app)
        request_id = str(uuid4())
        path = "/internal/v1/query"
        body = b"x" * 1025
        headers = sign_request(
            secret=REQUEST_SECRET,
            key_id="gateway",
            environment="staging",
            request_id=request_id,
            method="POST",
            path=path,
            body=body,
            timestamp=NOW,
        )
        headers["Content-Length"] = "1024"
        response = client.post(path, content=body, headers=headers)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "REQUEST_TOO_LARGE")
        verify_response(
            headers=response.headers,
            secrets_by_key_id={"worker": RESPONSE_SECRET},
            expected_environment="staging",
            request_id=request_id,
            status_code=413,
            body=response.content,
            now=NOW,
        )

    def test_query_rejects_revision_mismatch_then_submits_exact_ready_scope(self):
        payload = self.job_payload(
            question="What happened?",
            model="model:a",
            strategy="hybrid",
            maxSources=8,
        )
        missing, _, _ = self.signed_request("POST", "/internal/v1/query", payload)
        self.assertEqual(missing.status_code, 409)
        self.assertEqual(missing.json()["error"]["code"], "INDEX_NOT_READY")

        scope_model = NotebookScope.model_validate(self.scope)
        candidate = self.app.state.index_repository.build_candidate(
            FinalizedDocument(
                scope=scope_model,
                document_checksum=document_pages_checksum([]),
                pages=(),
            ),
            operation_id="op-test-ready-index",
        )
        committed = self.app.state.index_repository.commit_candidate(candidate)
        self.app.state.job_store.activate_index(
            scope_model,
            environment="staging",
            artifact_id=committed.artifact_id,
            now=NOW,
        )
        payload["requestId"] = str(uuid4())
        payload["idempotencyKey"] = f"idem-{uuid4()}"
        submitted, _, _ = self.signed_request("POST", "/internal/v1/query", payload)
        self.assertEqual(submitted.status_code, 202)
        job = submitted.json()["job"]
        self.assertEqual(job["operation"], "query")
        self.assertEqual(job["state"], "queued")
        self.assertNotIn("operationId", job)

        stored = self.app.state.job_store.get(job["jobId"], environment="staging")
        self.app.state.job_store.set_running(stored.operation_id, now=NOW)
        self.app.state.job_store.set_progress(
            stored.operation_id,
            0.30,
            progress_detail={
                "stage": "searching",
                "retrievalPlan": {
                    "originalQuestion": "What happened?",
                    "semanticQuery": "experiment outcome",
                    "bm25Terms": ["experiment", "outcome"],
                    "mode": "planned",
                },
            },
            now=NOW,
        )
        status, _, _ = self.signed_request("GET", f"/internal/v1/jobs/{job['jobId']}")
        self.assertEqual(status.status_code, 200)
        status_job = status.json()["job"]
        self.assertEqual(status_job["jobId"], job["jobId"])
        self.assertNotIn("result", status_job)
        self.assertNotIn("operationId", status_job)
        self.assertEqual(
            status_job["progressDetail"],
            {
                "stage": "searching",
                "retrievalPlan": {
                    "originalQuestion": "What happened?",
                    "semanticQuery": "experiment outcome",
                    "bm25Terms": ["experiment", "outcome"],
                    "mode": "planned",
                },
            },
        )

        changed_revision = dict(payload)
        changed_revision["requestId"] = str(uuid4())
        changed_revision["idempotencyKey"] = f"idem-{uuid4()}"
        changed_revision["scope"] = [{**self.scope, "contentRevision": "sha256:revision-b"}]
        stale, _, _ = self.signed_request("POST", "/internal/v1/query", changed_revision)
        self.assertEqual(stale.status_code, 409)

    def test_model_and_request_values_are_typed_and_allowlisted(self):
        unknown_model = self.job_payload(model="arbitrary:image", scope=[])
        response, _, _ = self.signed_request("POST", "/internal/v1/models/start", unknown_model)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "MODEL_NOT_APPROVED")

        command = self.job_payload(model="model:a", scope=[])
        command["command"] = "docker run --privileged arbitrary:image"
        invalid, _, _ = self.signed_request("POST", "/internal/v1/models/start", command)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["error"], {"code": "INVALID_REQUEST", "message": "Request body is invalid.", "retryable": False})
        self.assertNotIn("docker", invalid.text)

        allowed = self.job_payload(model="model:a", scope=[])
        accepted, _, _ = self.signed_request("POST", "/internal/v1/models/start", allowed)
        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(accepted.json()["job"]["operation"], "model_start")

    def test_ingest_batch_checksum_is_verified_before_submission(self):
        pages = [
            {
                "pageId": "page-a",
                "title": "Title",
                "text": "Normalized content",
                "sourceUrl": "/?page=page-a",
            }
        ]
        payload = self.job_payload(
            batchNumber=0,
            batchChecksum="sha256:" + "0" * 64,
            pages=pages,
        )
        rejected, _, _ = self.signed_request("POST", "/internal/v1/ingest/batches", payload)
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(rejected.json()["error"]["code"], "CHECKSUM_MISMATCH")

        payload["requestId"] = str(uuid4())
        payload["idempotencyKey"] = f"idem-{uuid4()}"
        payload["batchChecksum"] = ingest_pages_checksum(pages)
        accepted, _, _ = self.signed_request("POST", "/internal/v1/ingest/batches", payload)
        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(accepted.json()["job"]["operation"], "ingest_batch")

    def test_empty_notebook_can_finalize_without_an_ingest_batch(self):
        payload = self.job_payload(
            documentChecksum="sha256:" + hashlib.sha256(canonical_json([])).hexdigest(),
            pageCount=0,
            batchCount=0,
        )
        accepted, _, _ = self.signed_request("POST", "/internal/v1/ingest/finalize", payload)
        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(accepted.json()["job"]["operation"], "ingest_finalize")


if __name__ == "__main__":
    unittest.main()
