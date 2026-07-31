from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from novo_chat.runtime import (
    RuntimeConfigurationError,
    load_gateway_runtime,
    load_worker_runtime,
)


def write_secret(path: Path, marker: str) -> None:
    path.write_text((marker * 48)[:48], encoding="utf-8")
    os.chmod(path, 0o600)


class RuntimeConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def gateway_environment(self) -> dict[str, str]:
        secret_directory = self.root / "gateway-secrets"
        secret_directory.mkdir()
        for filename, marker in (
            ("integration", "i"),
            ("csrf", "c"),
            ("request", "q"),
            ("response", "r"),
        ):
            write_secret(secret_directory / filename, marker)
        state = self.root / "gateway-state"
        state.mkdir()
        return {
            "NOVO_CHAT_ENVIRONMENT": "staging",
            "NOVO_CHAT_BIND_HOST": "127.0.0.1",
            "NOVO_CHAT_PORT": "3181",
            "NOVO_CHAT_BASE_PATH": "/chat-staging",
            "NOVO_CHAT_PUBLIC_ORIGIN": "https://novo.example.test",
            "NOVO_INTEGRATION_API_URL": "http://127.0.0.1:3155/api/integrations/v1",
            "NOVO_INTEGRATION_SECRET_FILE": str(secret_directory / "integration"),
            "NOVO_CHAT_CSRF_SECRET_FILE": str(secret_directory / "csrf"),
            "NOVO_CHAT_WORKER_URL": "http://127.0.0.1:8196/internal/v1",
            "NOVO_CHAT_WORKER_SIGNING_KEY_FILE": str(secret_directory / "request"),
            "NOVO_CHAT_WORKER_RESPONSE_KEY_FILE": str(secret_directory / "response"),
            "NOVO_CHAT_WORKER_REQUEST_KEY_ID": "gateway-staging",
            "NOVO_CHAT_WORKER_RESPONSE_KEY_ID": "worker-staging",
            "NOVO_CHAT_INDEX_SCHEMA_VERSION": "novo-chat-index-v1",
            "NOVO_CHAT_JOB_DB": str(state / "jobs.sqlite3"),
            "NOVO_SESSION_COOKIE_NAME": "eln_session",
            "NOVO_HOME_PATH": "/",
            "NOVO_LOGIN_PATH": "/",
            "NOVO_LOGOUT_PATH": "/api/auth/logout",
            "NOVO_LOGIN_RETURN_PARAMETER": "returnTo",
        }

    def worker_environment(self) -> dict[str, str]:
        secret_directory = self.root / "worker-secrets"
        secret_directory.mkdir()
        write_secret(secret_directory / "request", "a")
        write_secret(secret_directory / "response", "b")
        state = self.root / "worker-state"
        state.mkdir()
        backend_file = self.root / "model-backends.json"
        backend_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "embedding": {
                        "baseUrl": "http://127.0.0.1:11434",
                        "model": "embedding-model",
                    },
                    "models": {
                        "qwen3.5:122b": {
                            "baseUrl": "http://127.0.0.1:8002",
                            "servedModel": "served-model",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        os.chmod(backend_file, 0o400)
        return {
            "NOVO_CHAT_ENVIRONMENT": "staging",
            "NOVO_CHAT_BIND_HOST": "127.0.0.1",
            "NOVO_CHAT_PORT": "8096",
            "NOVO_CHAT_WORKER_STATE_DB": str(state / "jobs.sqlite3"),
            "NOVO_CHAT_WORKER_STATE_DIR": str(state / "worker-runtime"),
            "NOVO_CHAT_WORKER_REQUEST_KEY_ID": "gateway-staging",
            "NOVO_CHAT_WORKER_REQUEST_KEY_FILE": str(secret_directory / "request"),
            "NOVO_CHAT_WORKER_RESPONSE_KEY_ID": "worker-staging",
            "NOVO_CHAT_WORKER_RESPONSE_KEY_FILE": str(secret_directory / "response"),
            "NOVO_CHAT_APPROVED_MODELS": "qwen3.5:122b",
            "NOVO_CHAT_INDEX_SCHEMA_VERSIONS": "novo-chat-index-v1",
            "NOVO_CHAT_MODEL_BACKENDS_FILE": str(backend_file),
            "NOVO_CHAT_MODELCTL_SOCKET": "/run/novo-chat/staging-modelctl.sock",
            "NOVO_CHAT_EXECUTOR_POLL_SECONDS": "0.5",
            "NOVO_CHAT_MODEL_START_TIMEOUT_SECONDS": "600",
            "NOVO_CHAT_JOB_RETENTION_SECONDS": "604800",
            "NOVO_CHAT_ORPHAN_GRACE_SECONDS": "3600",
            "NOVO_CHAT_DOCUMENT_STORAGE_QUOTA_BYTES": "53687091200",
            "NOVO_CHAT_INDEX_STORAGE_QUOTA_BYTES": "53687091200",
        }

    def test_staging_gateway_contract_loads(self) -> None:
        runtime = load_gateway_runtime(self.gateway_environment())
        self.assertEqual((runtime.host, runtime.port), ("127.0.0.1", 3181))
        self.assertEqual(runtime.settings.environment, "staging")
        self.assertEqual(runtime.settings.novo_export_page_limit, 8)
        self.assertEqual(runtime.settings.ingest_batch_max_bytes, 24 * 1024 * 1024)
        self.assertEqual(runtime.settings.gateway_max_request_bytes, 64 * 1024)
        self.assertEqual(runtime.settings.job_ownership_ttl_s, 604800)

    def test_gateway_retention_and_ingest_byte_cap_are_configurable(self) -> None:
        environment = self.gateway_environment()
        environment["NOVO_CHAT_JOB_OWNERSHIP_TTL_S"] = "172800"
        environment["NOVO_CHAT_INGEST_BATCH_MAX_BYTES"] = "1048576"
        environment["NOVO_CHAT_GATEWAY_MAX_REQUEST_BYTES"] = "32768"
        runtime = load_gateway_runtime(environment)
        self.assertEqual(runtime.settings.job_ownership_ttl_s, 172_800)
        self.assertEqual(runtime.settings.ingest_batch_max_bytes, 1_048_576)
        self.assertEqual(runtime.settings.gateway_max_request_bytes, 32_768)

    def test_staging_worker_contract_loads(self) -> None:
        runtime = load_worker_runtime(self.worker_environment(), require_socket=False)
        self.assertEqual((runtime.host, runtime.port), ("127.0.0.1", 8096))
        self.assertEqual(runtime.config.approved_models, ("qwen3.5:122b",))
        self.assertEqual(
            runtime.controller.socket_path,
            Path("/run/novo-chat/staging-modelctl.sock"),
        )
        self.assertEqual(runtime.model_backend.approved_models, ("qwen3.5:122b",))
        self.assertEqual(runtime.config.job_retention_seconds, 604800)
        self.assertEqual(runtime.config.orphan_grace_seconds, 3600)
        self.assertEqual(runtime.config.document_storage_quota_bytes, 50 * 1024**3)
        self.assertEqual(runtime.config.index_storage_quota_bytes, 50 * 1024**3)

    def test_development_environment_is_rejected(self) -> None:
        environment = self.gateway_environment()
        environment["NOVO_CHAT_ENVIRONMENT"] = "development"
        with self.assertRaisesRegex(RuntimeConfigurationError, "staging or production"):
            load_gateway_runtime(environment)

    def test_non_loopback_bind_is_rejected(self) -> None:
        environment = self.worker_environment()
        environment["NOVO_CHAT_BIND_HOST"] = "0.0.0.0"
        with self.assertRaisesRegex(RuntimeConfigurationError, "literal IPv4 loopback"):
            load_worker_runtime(environment, require_socket=False)

    def test_environment_role_port_mismatch_is_rejected(self) -> None:
        environment = self.gateway_environment()
        environment["NOVO_CHAT_PORT"] = "3180"
        with self.assertRaisesRegex(RuntimeConfigurationError, "must be 3181"):
            load_gateway_runtime(environment)

    def test_remote_upstream_is_rejected(self) -> None:
        environment = self.gateway_environment()
        environment["NOVO_CHAT_WORKER_URL"] = "http://compute.example.test:8196/internal/v1"
        with self.assertRaisesRegex(RuntimeConfigurationError, "127.0.0.1:8196"):
            load_gateway_runtime(environment)

    def test_public_origin_rejects_invalid_port(self) -> None:
        environment = self.gateway_environment()
        environment["NOVO_CHAT_PUBLIC_ORIGIN"] = "https://novo.example.test:not-a-port"
        with self.assertRaisesRegex(RuntimeConfigurationError, "invalid port"):
            load_gateway_runtime(environment)

    def test_world_readable_secret_is_rejected(self) -> None:
        environment = self.worker_environment()
        request_file = Path(environment["NOVO_CHAT_WORKER_REQUEST_KEY_FILE"])
        os.chmod(request_file, 0o644)
        with self.assertRaisesRegex(RuntimeConfigurationError, "unsafe permissions"):
            load_worker_runtime(environment, require_socket=False)

    def test_secret_symlink_is_rejected(self) -> None:
        environment = self.gateway_environment()
        source = Path(environment["NOVO_CHAT_CSRF_SECRET_FILE"])
        link = source.with_name("csrf-link")
        link.symlink_to(source)
        environment["NOVO_CHAT_CSRF_SECRET_FILE"] = str(link)
        with self.assertRaisesRegex(RuntimeConfigurationError, "regular secret file"):
            load_gateway_runtime(environment)

    def test_same_direction_keys_are_rejected(self) -> None:
        environment = self.worker_environment()
        request_file = Path(environment["NOVO_CHAT_WORKER_REQUEST_KEY_FILE"])
        response_file = Path(environment["NOVO_CHAT_WORKER_RESPONSE_KEY_FILE"])
        response_file.write_bytes(request_file.read_bytes())
        os.chmod(response_file, 0o600)
        with self.assertRaisesRegex(RuntimeConfigurationError, "distinct values"):
            load_worker_runtime(environment, require_socket=False)

    def test_unresolved_model_placeholder_is_rejected(self) -> None:
        environment = self.worker_environment()
        environment["NOVO_CHAT_APPROVED_MODELS"] = "@STAGING_APPROVED_MODEL@"
        with self.assertRaisesRegex(RuntimeConfigurationError, "unresolved template placeholder"):
            load_worker_runtime(environment, require_socket=False)

    def test_model_id_uses_protocol_safe_token_alphabet(self) -> None:
        environment = self.worker_environment()
        environment["NOVO_CHAT_APPROVED_MODELS"] = "unsafe+model"
        with self.assertRaisesRegex(RuntimeConfigurationError, "invalid identifier"):
            load_worker_runtime(environment, require_socket=False)

    def test_approved_models_must_match_backend_document(self) -> None:
        environment = self.worker_environment()
        environment["NOVO_CHAT_APPROVED_MODELS"] = "gpt-oss:120b"
        with self.assertRaisesRegex(RuntimeConfigurationError, "exactly match"):
            load_worker_runtime(environment, require_socket=False)

    def test_model_backend_config_must_be_loopback_only(self) -> None:
        environment = self.worker_environment()
        backend_file = Path(environment["NOVO_CHAT_MODEL_BACKENDS_FILE"])
        document = json.loads(backend_file.read_text(encoding="utf-8"))
        document["embedding"]["baseUrl"] = "https://models.example.test"
        os.chmod(backend_file, 0o600)
        backend_file.write_text(json.dumps(document), encoding="utf-8")
        os.chmod(backend_file, 0o400)
        with self.assertRaisesRegex(RuntimeConfigurationError, "is invalid"):
            load_worker_runtime(environment, require_socket=False)

    def test_model_backend_config_rejects_unresolved_placeholder(self) -> None:
        environment = self.worker_environment()
        backend_file = Path(environment["NOVO_CHAT_MODEL_BACKENDS_FILE"])
        os.chmod(backend_file, 0o600)
        backend_file.write_text(
            backend_file.read_text(encoding="utf-8").replace("11434", "@EMBEDDING_PORT@"),
            encoding="utf-8",
        )
        os.chmod(backend_file, 0o400)
        with self.assertRaisesRegex(RuntimeConfigurationError, "unresolved template placeholder"):
            load_worker_runtime(environment, require_socket=False)

    def test_job_retention_must_be_positive_integer_seconds(self) -> None:
        environment = self.worker_environment()
        environment["NOVO_CHAT_JOB_RETENTION_SECONDS"] = "0"
        with self.assertRaisesRegex(RuntimeConfigurationError, "must be positive"):
            load_worker_runtime(environment, require_socket=False)

    def test_orphan_grace_cannot_outlive_active_index_retention(self) -> None:
        environment = self.worker_environment()
        environment["NOVO_CHAT_ORPHAN_GRACE_SECONDS"] = "604801"
        with self.assertRaisesRegex(RuntimeConfigurationError, "must not exceed"):
            load_worker_runtime(environment, require_socket=False)


if __name__ == "__main__":
    unittest.main()
