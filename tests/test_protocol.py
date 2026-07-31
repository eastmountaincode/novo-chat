from __future__ import annotations

import unittest

from pydantic import ValidationError

from novo_chat.protocol import (
    Citation,
    InMemoryReplayGuard,
    ModelJobRequest,
    NotebookScope,
    PageDocument,
    ProtocolError,
    RebuildRequest,
    canonical_json,
    canonicalize_query,
    sign_request,
    sign_response,
    verify_request,
    verify_response,
)


SECRET = b"0123456789abcdef0123456789abcdef"
REQUEST_ID = "123e4567-e89b-42d3-a456-426614174000"


class CanonicalProtocolTests(unittest.TestCase):
    def test_canonical_json_and_query_are_deterministic(self):
        self.assertEqual(canonical_json({"z": 1, "a": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"}), b'{"a":"caf\xc3\xa9","z":1}')
        self.assertEqual(canonicalize_query("z=last&a=hello+world&a=first"), "a=first&a=hello%20world&z=last")

    def test_request_signature_round_trip_tamper_and_replay(self):
        body = canonical_json({"hello": "world"})
        headers = sign_request(
            secret=SECRET,
            key_id="gateway-2026-01",
            environment="staging",
            request_id=REQUEST_ID,
            method="POST",
            path="/internal/v1/query",
            query="z=last&a=first",
            body=body,
            timestamp=1_700_000_000,
            nonce="fixed-nonce-12345678",
        )
        self.assertEqual(
            headers["X-Novo-Chat-Signature"],
            "f7fcb8311d9f67e8265c5f7a05ee4325b0d01f56525acf99acc952f584e05e9c",
        )
        replay_guard = InMemoryReplayGuard()
        verified = verify_request(
            headers=headers,
            secrets_by_key_id={"gateway-2026-01": SECRET},
            expected_environment="staging",
            method="POST",
            path="/internal/v1/query",
            query="a=first&z=last",
            body=body,
            replay_guard=replay_guard,
            now=1_700_000_030,
        )
        self.assertEqual(verified.request_id, REQUEST_ID)
        with self.assertRaisesRegex(ProtocolError, "already been used"):
            verify_request(
                headers=headers,
                secrets_by_key_id={"gateway-2026-01": SECRET},
                expected_environment="staging",
                method="POST",
                path="/internal/v1/query",
                query="a=first&z=last",
                body=body,
                replay_guard=replay_guard,
                now=1_700_000_031,
            )
        with self.assertRaises(ProtocolError) as tampered:
            verify_request(
                headers=headers,
                secrets_by_key_id={"gateway-2026-01": SECRET},
                expected_environment="staging",
                method="POST",
                path="/internal/v1/query",
                query="a=first&z=last",
                body=body + b" ",
                now=1_700_000_030,
            )
        self.assertEqual(tampered.exception.code, "AUTHENTICATION_FAILED")

    def test_wrong_environment_and_stale_request_are_rejected(self):
        headers = sign_request(
            secret=SECRET,
            key_id="gateway",
            environment="production",
            request_id=REQUEST_ID,
            method="GET",
            path="/internal/v1/health",
            timestamp=1_700_000_000,
            nonce="fixed-nonce-12345678",
        )
        with self.assertRaises(ProtocolError) as wrong_environment:
            verify_request(
                headers=headers,
                secrets_by_key_id={"gateway": SECRET},
                expected_environment="staging",
                method="GET",
                path="/internal/v1/health",
                now=1_700_000_000,
            )
        self.assertEqual(wrong_environment.exception.code, "AUTHENTICATION_FAILED")
        with self.assertRaises(ProtocolError) as stale:
            verify_request(
                headers=headers,
                secrets_by_key_id={"gateway": SECRET},
                expected_environment="production",
                method="GET",
                path="/internal/v1/health",
                now=1_700_000_061,
            )
        self.assertEqual(stale.exception.code, "REQUEST_EXPIRED")

    def test_response_is_bound_to_request_status_and_body(self):
        body = canonical_json({"ok": True})
        headers = sign_response(
            secret=SECRET,
            key_id="worker",
            environment="staging",
            request_id=REQUEST_ID,
            status_code=202,
            body=body,
            timestamp=1_700_000_000,
        )
        verified = verify_response(
            headers=headers,
            secrets_by_key_id={"worker": SECRET},
            expected_environment="staging",
            request_id=REQUEST_ID,
            status_code=202,
            body=body,
            now=1_700_000_020,
        )
        self.assertEqual(verified.status_code, 202)
        for changed_status, changed_body in ((200, body), (202, b'{"ok":false}')):
            with self.assertRaises(ProtocolError):
                verify_response(
                    headers=headers,
                    secrets_by_key_id={"worker": SECRET},
                    expected_environment="staging",
                    request_id=REQUEST_ID,
                    status_code=changed_status,
                    body=changed_body,
                    now=1_700_000_020,
                )


class ProtocolModelTests(unittest.TestCase):
    def test_source_urls_are_same_origin_browser_paths(self):
        self.assertEqual(
            Citation(notebookId="n1", pageId="p1", sourceUrl="/?page=p1").source_url,
            "/?page=p1",
        )
        for unsafe in (
            "javascript:alert(1)",
            "https://attacker.invalid/page",
            "//attacker.invalid/page",
            "/\\attacker.invalid/page",
            "/%5cattacker.invalid/page",
            "/page%2f..%2fsecret",
            "/page\nnext",
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValidationError):
                Citation(notebookId="n1", pageId="p1", sourceUrl=unsafe)
            with self.subTest(unsafe=unsafe), self.assertRaises(ValidationError):
                PageDocument(pageId="p1", sourceUrl=unsafe)

    def test_scope_uses_exact_wire_field_names(self):
        scope = NotebookScope(
            notebook_id="notebook-a",
            content_revision="sha256:abc",
            index_schema_version="index-v1",
        )
        self.assertEqual(
            scope.model_dump(mode="json", by_alias=True),
            {
                "notebookId": "notebook-a",
                "contentRevision": "sha256:abc",
                "indexSchemaVersion": "index-v1",
            },
        )

    def test_wildcards_duplicate_scope_and_model_scope_are_rejected(self):
        with self.assertRaises(ValidationError):
            NotebookScope(notebookId="*", contentRevision="rev", indexSchemaVersion="v1")
        scope = {"notebookId": "n1", "contentRevision": "rev", "indexSchemaVersion": "v1"}
        with self.assertRaises(ValidationError):
            RebuildRequest(
                requestId=REQUEST_ID,
                idempotencyKey="idempotency-1",
                actorUserId="user-1",
                scope=[scope, scope],
            )
        with self.assertRaises(ValidationError):
            ModelJobRequest(
                requestId=REQUEST_ID,
                idempotencyKey="idempotency-2",
                actorUserId="user-1",
                scope=[scope],
                model="model:a",
            )


if __name__ == "__main__":
    unittest.main()
