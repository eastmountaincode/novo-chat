from __future__ import annotations

import configparser
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from novo_chat.model_backend import ModelBackendDocument


ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "deploy"
PLACEHOLDER = re.compile(r"@[A-Z][A-Z0-9_]*@")


def read(relative: str) -> str:
    return (DEPLOY / relative).read_text(encoding="utf-8")


def env_file(relative: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in read(relative).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise AssertionError(f"invalid or duplicate environment line in {relative}: {raw_line}")
        values[key] = value
    return values


class DeploymentTemplateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ssh"), "OpenSSH client is unavailable")
    def test_tunnel_templates_parse_with_openssh(self) -> None:
        replacements = {
            "@JUMP_HOST@": "jump.example.test",
            "@JUMP_TUNNEL_USER@": "jump-tunnel",
            "@GATEWAY_SSH_HOST@": "gateway.example.test",
            "@GATEWAY_SSH_HOST_KEY_ALIAS@": "gateway-host-key",
            "@STAGING_TARGET_TUNNEL_USER@": "staging-tunnel",
            "@PRODUCTION_TARGET_TUNNEL_USER@": "production-tunnel",
        }
        for relative, host in (
            ("compute-host/tunnel-staging.conf.template", "novo-chat-gateway-staging"),
            ("compute-host/tunnel-production.conf.template", "novo-chat-gateway-production"),
        ):
            rendered = PLACEHOLDER.sub(lambda match: replacements[match.group(0)], read(relative))
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "ssh_config"
                path.write_text(rendered, encoding="utf-8")
                result = subprocess.run(
                    [shutil.which("ssh") or "ssh", "-G", "-F", str(path), host],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            self.assertEqual(result.returncode, 0, f"{relative}: {result.stderr}")

    def test_container_package_contains_only_distributed_package(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertNotIn("COPY novo_rag", dockerfile)
        self.assertIn('include = ["novo_chat*"]', project)
        self.assertNotIn('"novo_rag*"', project)
        self.assertNotIn("novo_rag", manifest)

    def test_json_templates_are_structurally_valid(self) -> None:
        for relative in (
            "compute-host/model-backends-staging.json.template",
            "compute-host/model-backends-production.json.template",
        ):
            rendered = PLACEHOLDER.sub("1", read(relative))
            document = ModelBackendDocument.model_validate(json.loads(rendered))
            self.assertEqual(document.version, 1)
            self.assertEqual(tuple(document.models), ("1",))
        for relative in (
            "compute-host/models-staging.json.template",
            "compute-host/models-production.json.template",
        ):
            document = json.loads(PLACEHOLDER.sub("1", read(relative)))
            self.assertEqual(document["version"], 1)
            self.assertEqual(tuple(document["models"]), ("1",))
        inventory = json.loads(
            read("compute-host/modelctl-exclusive-containers.json.template")
            .replace("@STAGING_TEST_MODEL_CONTAINER@", "staging-model")
            .replace("@EXISTING_PRODUCTION_MODEL_CONTAINER@", "production-model")
        )
        self.assertEqual(inventory["version"], 1)
        self.assertEqual(inventory["containers"], ["staging-model", "production-model"])

    def test_systemd_templates_have_parseable_sections(self) -> None:
        for relative, required_section in (
            ("gateway-host/novo-chat-gateway@.service.template", "Service"),
            ("compute-host/novo-chat-worker@.service.template", "Service"),
            ("compute-host/novo-chat-tunnel@.service.template", "Service"),
            ("compute-host/novo-chat-modelctl@.service.template", "Service"),
            ("compute-host/novo-chat-modelctl@.socket", "Socket"),
        ):
            parser = configparser.RawConfigParser(strict=False, interpolation=None)
            parser.optionxform = str
            parser.read_string(read(relative), source=relative)
            self.assertIn("Unit", parser.sections(), relative)
            self.assertIn(required_section, parser.sections(), relative)

    def test_role_environment_ports_and_state_are_isolated(self) -> None:
        staging_gateway = env_file("gateway-host/gateway-staging.env.template")
        production_gateway = env_file("gateway-host/gateway-production.env.template")
        staging_worker = env_file("compute-host/worker-staging.env.template")
        production_worker = env_file("compute-host/worker-production.env.template")

        self.assertEqual(staging_gateway["NOVO_CHAT_PORT"], "3181")
        self.assertEqual(production_gateway["NOVO_CHAT_PORT"], "3180")
        self.assertEqual(staging_worker["NOVO_CHAT_PORT"], "8096")
        self.assertEqual(production_worker["NOVO_CHAT_PORT"], "8095")
        self.assertEqual(staging_gateway["NOVO_EXPORT_PAGE_LIMIT"], "8")
        self.assertEqual(production_gateway["NOVO_EXPORT_PAGE_LIMIT"], "8")
        self.assertEqual(staging_worker["NOVO_CHAT_JOB_RETENTION_SECONDS"], "604800")
        self.assertEqual(production_worker["NOVO_CHAT_JOB_RETENTION_SECONDS"], "604800")
        self.assertEqual(staging_worker["NOVO_CHAT_ORPHAN_GRACE_SECONDS"], "3600")
        self.assertEqual(production_worker["NOVO_CHAT_ORPHAN_GRACE_SECONDS"], "3600")
        self.assertEqual(staging_worker["NOVO_CHAT_DOCUMENT_STORAGE_QUOTA_BYTES"], "53687091200")
        self.assertEqual(production_worker["NOVO_CHAT_DOCUMENT_STORAGE_QUOTA_BYTES"], "53687091200")
        self.assertEqual(staging_worker["NOVO_CHAT_INDEX_STORAGE_QUOTA_BYTES"], "53687091200")
        self.assertEqual(production_worker["NOVO_CHAT_INDEX_STORAGE_QUOTA_BYTES"], "53687091200")
        self.assertEqual(staging_gateway["NOVO_CHAT_INGEST_BATCH_MAX_BYTES"], "25165824")
        self.assertEqual(production_gateway["NOVO_CHAT_INGEST_BATCH_MAX_BYTES"], "25165824")
        self.assertEqual(staging_gateway["NOVO_CHAT_GATEWAY_MAX_REQUEST_BYTES"], "65536")
        self.assertEqual(production_gateway["NOVO_CHAT_GATEWAY_MAX_REQUEST_BYTES"], "65536")
        self.assertEqual(staging_gateway["NOVO_CHAT_JOB_OWNERSHIP_TTL_S"], "604800")
        self.assertEqual(production_gateway["NOVO_CHAT_JOB_OWNERSHIP_TTL_S"], "604800")
        self.assertEqual(staging_gateway["NOVO_CHAT_PUBLIC_ORIGIN"], "@STAGING_PUBLIC_ORIGIN@")
        self.assertEqual(production_gateway["NOVO_CHAT_PUBLIC_ORIGIN"], "https://@PUBLIC_HOST@")
        for values in (staging_gateway, production_gateway, staging_worker, production_worker):
            self.assertEqual(values["NOVO_CHAT_BIND_HOST"], "127.0.0.1")
        self.assertNotEqual(
            staging_gateway["NOVO_CHAT_JOB_DB"], production_gateway["NOVO_CHAT_JOB_DB"]
        )
        self.assertNotEqual(
            staging_worker["NOVO_CHAT_WORKER_STATE_DB"],
            production_worker["NOVO_CHAT_WORKER_STATE_DB"],
        )

    def test_every_secret_setting_is_a_mounted_file_path(self) -> None:
        for relative in (
            "gateway-host/gateway-staging.env.template",
            "gateway-host/gateway-production.env.template",
            "compute-host/worker-staging.env.template",
            "compute-host/worker-production.env.template",
        ):
            values = env_file(relative)
            secret_settings = {
                key: value
                for key, value in values.items()
                if key.endswith("_SECRET_FILE")
                or key.endswith("_KEY_FILE")
                or key == "NOVO_INTEGRATION_SECRET_FILE"
            }
            self.assertTrue(secret_settings, relative)
            for setting, value in secret_settings.items():
                self.assertTrue(value.startswith("/run/secrets/"), f"{relative}: {setting}")

    def test_gateway_container_dispatch_and_mounts(self) -> None:
        unit = read("gateway-host/novo-chat-gateway@.service.template")
        self.assertIn("@NOVO_CHAT_IMAGE_DIGEST@ gateway", unit)
        self.assertIn("--user=10001:10001", unit)
        self.assertIn("/var/lib/novo-chat-gateway/%i", unit)
        self.assertIn("dst=/run/secrets,readonly", unit)
        self.assertIn("--network host", unit)
        self.assertIn("--memory=768m", unit)
        self.assertIn("--memory-swap=768m", unit)
        self.assertIn("MemoryMax=1G", unit)
        self.assertIn("MemorySwapMax=0", unit)
        self.assertNotIn("docker.sock", unit)
        self.assertNotIn("--privileged", unit)

    def test_worker_has_only_narrow_controller_privilege(self) -> None:
        unit = read("compute-host/novo-chat-worker@.service.template")
        self.assertIn("@NOVO_CHAT_IMAGE_DIGEST@ worker", unit)
        self.assertIn("--user=10001:10001", unit)
        self.assertIn("--group-add=@MODELCTL_SOCKET_GID@", unit)
        self.assertIn("src=/run/novo-chat/%i-modelctl.sock", unit)
        self.assertIn("src=/etc/novo-chat/model-backends-%i.json", unit)
        self.assertIn("dst=/run/config/model-backends.json,readonly", unit)
        self.assertIn("/var/lib/novo-chat-worker/%i", unit)
        self.assertIn("--memory=16g", unit)
        self.assertIn("--memory-swap=16g", unit)
        self.assertIn("MemoryMax=17G", unit)
        self.assertIn("MemorySwapMax=0", unit)
        self.assertNotIn("docker.sock", unit)
        self.assertNotIn("--privileged", unit)

    def test_model_controller_runtime_directory_is_group_traversable(self) -> None:
        tmpfiles = read("compute-host/novo-chat.tmpfiles.template")
        socket_unit = read("compute-host/novo-chat-modelctl@.socket")
        service_unit = read("compute-host/novo-chat-modelctl@.service.template")
        self.assertIn("d /run/novo-chat 0750 root novo-chat-modelctl -", tmpfiles)
        self.assertIn("SocketGroup=novo-chat-modelctl", socket_unit)
        self.assertIn("SocketMode=0660", socket_unit)
        self.assertIn(
            "--inventory /etc/novo-chat/modelctl-exclusive-containers.json",
            service_unit,
        )
        self.assertIn("ReadWritePaths=/run/novo-chat", service_unit)

    def test_tunnels_are_pinned_loopback_forwards(self) -> None:
        staging = read("compute-host/tunnel-staging.conf.template")
        production = read("compute-host/tunnel-production.conf.template")
        self.assertIn("RemoteForward 127.0.0.1:8196 127.0.0.1:8096", staging)
        self.assertIn("RemoteForward 127.0.0.1:8195 127.0.0.1:8095", production)
        for config in (staging, production):
            self.assertIn("StrictHostKeyChecking yes", config)
            self.assertIn("ExitOnForwardFailure yes", config)
            self.assertNotRegex(config, r"(?m)^\s*StrictHostKeyChecking\s+accept-new\s*$")

    def test_apache_routes_and_private_integration_exclusion(self) -> None:
        common = read("gateway-host/apache-novo-chat-common.conf.template")
        staging = read("gateway-host/apache-novo-chat-staging.conf.template")
        production = read("gateway-host/apache-novo-chat.conf.template")
        self.assertIn('ProxyPass "/api/integrations/v1" "!"', common)
        self.assertIn("Require all denied", common)
        self.assertIn("127.0.0.1:3181/chat-staging/", staging)
        self.assertIn("127.0.0.1:3180/chat/", production)
        self.assertIn("LimitRequestBody 65536", staging)
        self.assertIn("LimitRequestBody 65536", production)
        for snippet in (common, staging, production):
            self.assertNotIn("SSLCertificate", snippet)
            self.assertNotIn("<VirtualHost", snippet)

    def test_staging_caddy_mux_is_loopback_only_and_preserves_one_origin(self) -> None:
        caddy = read("gateway-host/Caddyfile-staging-mux.template")
        self.assertIn("http://127.0.0.1:3182", caddy)
        self.assertNotIn("0.0.0.0", caddy)
        self.assertIn("admin off", caddy)
        self.assertRegex(caddy, r"path /api/integrations/v1 /api/integrations/v1/\*")
        self.assertRegex(caddy, r"handle @privateIntegration \{[\s\S]*?respond 404")
        self.assertRegex(caddy, r"path /chat-staging /chat-staging/\*")
        self.assertIn("reverse_proxy 127.0.0.1:3181", caddy)
        self.assertIn("reverse_proxy 127.0.0.1:3155", caddy)
        self.assertIn("max_size 64KB", caddy)
        self.assertIn('X-Content-Type-Options "nosniff"', caddy)
        self.assertIn('Referrer-Policy "same-origin"', caddy)
        self.assertIn('X-Frame-Options "SAMEORIGIN"', caddy)

    def test_public_templates_do_not_contain_site_accounts_or_hostnames(self) -> None:
        forbidden = re.compile(
            r"(?i)(?:/Users/|/home/[a-z0-9._-]+|/mnt/|(?:[a-z0-9-]+\.)+(?:edu|org)\b)"
        )
        for path in DEPLOY.rglob("*"):
            if (
                not path.is_file()
                or path.name.startswith("._")
                or path.name.startswith("PHASE0_AUDIT_")
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8")), str(path))

    def test_placeholder_inventory_is_intentional(self) -> None:
        allowed = {
            "@EXISTING_PRODUCTION_MODEL_CONTAINER@",
            "@GATEWAY_SSH_HOST@",
            "@GATEWAY_SSH_HOST_KEY_ALIAS@",
            "@JUMP_HOST@",
            "@JUMP_TUNNEL_USER@",
            "@MODELCTL_SOCKET_GID@",
            "@NOVO_CHAT_IMAGE_DIGEST@",
            "@PRODUCTION_APPROVED_MODEL@",
            "@PRODUCTION_EMBEDDING_MODEL@",
            "@PRODUCTION_EMBEDDING_PORT@",
            "@PRODUCTION_GENERATION_PORT@",
            "@PRODUCTION_JUMP_PUBLIC_KEY@",
            "@PRODUCTION_SERVED_MODEL@",
            "@PRODUCTION_TARGET_PUBLIC_KEY@",
            "@PRODUCTION_TARGET_TUNNEL_USER@",
            "@PUBLIC_HOST@",
            "@STAGING_ALLOWED_CIDR@",
            "@STAGING_APPROVED_MODEL@",
            "@STAGING_EMBEDDING_MODEL@",
            "@STAGING_EMBEDDING_PORT@",
            "@STAGING_GENERATION_PORT@",
            "@STAGING_JUMP_PUBLIC_KEY@",
            "@STAGING_PUBLIC_ORIGIN@",
            "@STAGING_SERVED_MODEL@",
            "@STAGING_TARGET_PUBLIC_KEY@",
            "@STAGING_TARGET_TUNNEL_USER@",
            "@STAGING_TEST_MODEL_CONTAINER@",
        }
        found: set[str] = set()
        for path in DEPLOY.rglob("*"):
            if (
                not path.is_file()
                or path.name.startswith("._")
                or path.name.startswith("PHASE0_AUDIT_")
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            found.update(PLACEHOLDER.findall(path.read_text(encoding="utf-8")))
        self.assertEqual(found, allowed)


if __name__ == "__main__":
    unittest.main()
