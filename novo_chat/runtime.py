"""Fail-closed deployment entrypoints for the gateway and compute worker.

The application factories remain convenient for unit tests. This module is the
only supported container command: it requires an explicit deployment
environment, checks loopback/port boundaries, reads secret files without
following symlinks, and then constructs the appropriate app.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit

import uvicorn

from .gateway.config import GatewaySettings
from .model_backend import HttpModelBackend, load_model_backend_document
from .model_controller import UnixSocketModelController
from .worker import WorkerConfig


DEPLOYMENT_PORTS = {
    "gateway": {"staging": 3181, "production": 3180},
    "worker": {"staging": 8096, "production": 8095},
}
NOVO_PORTS = {"staging": 3155, "production": 3148}
REVERSE_FORWARD_PORTS = {"staging": 8196, "production": 8195}
BASE_PATHS = {"staging": "/chat-staging", "production": "/chat"}

_PLACEHOLDER = re.compile(r"@[A-Z][A-Z0-9_]*@")
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SCHEMA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_SECRET_BYTES = 64 * 1024


class RuntimeConfigurationError(ValueError):
    """A safe-to-display deployment configuration failure."""


@dataclass(frozen=True, slots=True)
class GatewayRuntime:
    host: str
    port: int
    settings: GatewaySettings


@dataclass(frozen=True, slots=True)
class WorkerRuntime:
    host: str
    port: int
    config: WorkerConfig
    controller: UnixSocketModelController
    model_backend: HttpModelBackend


def _value(environment: Mapping[str, str], name: str, *, default: str | None = None) -> str:
    raw = environment.get(name, default)
    if raw is None or not str(raw).strip():
        raise RuntimeConfigurationError(f"{name} is required")
    value = str(raw).strip()
    if _PLACEHOLDER.search(value):
        raise RuntimeConfigurationError(f"{name} contains an unresolved template placeholder")
    return value


def _environment(environment: Mapping[str, str]) -> str:
    value = _value(environment, "NOVO_CHAT_ENVIRONMENT")
    if value not in {"staging", "production"}:
        raise RuntimeConfigurationError("NOVO_CHAT_ENVIRONMENT must be staging or production")
    return value


def _bind(environment: Mapping[str, str], *, role: str, deployment: str) -> tuple[str, int]:
    host = _value(environment, "NOVO_CHAT_BIND_HOST")
    if host != "127.0.0.1":
        raise RuntimeConfigurationError("NOVO_CHAT_BIND_HOST must be the literal IPv4 loopback 127.0.0.1")
    raw_port = _value(environment, "NOVO_CHAT_PORT")
    try:
        port = int(raw_port, 10)
    except ValueError as exc:
        raise RuntimeConfigurationError("NOVO_CHAT_PORT must be an integer") from exc
    expected = DEPLOYMENT_PORTS[role][deployment]
    if port != expected:
        raise RuntimeConfigurationError(
            f"NOVO_CHAT_PORT must be {expected} for the {deployment} {role}"
        )
    return host, port


def _absolute_path(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_value(environment, name))
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeConfigurationError(f"{name} must be an absolute path without parent traversal")
    return path


def _read_secret(path: Path, *, setting: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeConfigurationError(f"{setting} does not name a readable regular secret file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeConfigurationError(f"{setting} must name a regular file")
        if metadata.st_uid not in {0, os.geteuid()}:
            raise RuntimeConfigurationError(f"{setting} must be owned by root or the service user")
        # Owner write is permitted for managed rotation; group write/execute,
        # every world bit, and owner execute are forbidden. Group read supports
        # the root:10001 0440 deployment files.
        if metadata.st_mode & 0o137:
            raise RuntimeConfigurationError(f"{setting} has unsafe permissions")
        if metadata.st_nlink != 1:
            raise RuntimeConfigurationError(f"{setting} must not be hard-linked")
        if metadata.st_size > _MAX_SECRET_BYTES:
            raise RuntimeConfigurationError(f"{setting} is too large")
        chunks: list[bytes] = []
        remaining = _MAX_SECRET_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(4096, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks).strip()
    finally:
        os.close(descriptor)
    if len(value) < 32:
        raise RuntimeConfigurationError(f"{setting} must contain at least 32 bytes")
    if len(value) > _MAX_SECRET_BYTES or b"\x00" in value:
        raise RuntimeConfigurationError(f"{setting} contains invalid secret data")
    return value


def _state_target(path: Path, *, setting: str, directory: bool = False) -> None:
    if path.is_symlink():
        raise RuntimeConfigurationError(f"{setting} must not be a symbolic link")
    if path.exists():
        if directory and not path.is_dir():
            raise RuntimeConfigurationError(f"{setting} must name a directory")
        if not directory and not path.is_file():
            raise RuntimeConfigurationError(f"{setting} must name a regular file when it already exists")
    ancestor = path if path.exists() and path.is_dir() else path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeConfigurationError(f"{setting} must be under a writable state mount")


def _readonly_config(path: Path, *, setting: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeConfigurationError(f"{setting} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeConfigurationError(f"{setting} must name a regular file")
    if metadata.st_uid not in {0, os.geteuid()}:
        raise RuntimeConfigurationError(f"{setting} must be owned by root or the service user")
    if metadata.st_mode & 0o022:
        raise RuntimeConfigurationError(f"{setting} must not be group- or world-writable")
    if metadata.st_size < 2 or metadata.st_size > 1024 * 1024:
        raise RuntimeConfigurationError(f"{setting} has an invalid size")
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeConfigurationError(f"{setting} is unreadable") from exc
    if _PLACEHOLDER.search(contents):
        raise RuntimeConfigurationError(f"{setting} contains an unresolved template placeholder")


def _identifier(value: str, *, setting: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value):
        raise RuntimeConfigurationError(f"{setting} contains an invalid identifier")
    return value


def _csv_identifiers(
    environment: Mapping[str, str],
    name: str,
    *,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    values = tuple(part.strip() for part in _value(environment, name).split(",") if part.strip())
    if not values:
        raise RuntimeConfigurationError(f"{name} must contain at least one identifier")
    if len(values) != len(set(values)):
        raise RuntimeConfigurationError(f"{name} must not contain duplicate identifiers")
    for value in values:
        _identifier(value, setting=name, pattern=pattern)
    return values


def _number(
    environment: Mapping[str, str],
    name: str,
    *,
    default: str,
    integer: bool = False,
) -> int | float:
    raw = _value(environment, name, default=default)
    try:
        value = int(raw, 10) if integer else float(raw)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{name} must be numeric") from exc
    if not integer and not math.isfinite(float(value)):
        raise RuntimeConfigurationError(f"{name} must be finite")
    if value <= 0:
        raise RuntimeConfigurationError(f"{name} must be positive")
    return value


def _nonnegative_integer(environment: Mapping[str, str], name: str, *, default: str) -> int:
    raw = _value(environment, name, default=default)
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise RuntimeConfigurationError(f"{name} must not be negative")
    return value


def _loopback_url(value: str, *, setting: str, port: int, path: str) -> str:
    parsed = urlsplit(value)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{setting} contains an invalid port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed_port != port
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != path.rstrip("/")
    ):
        raise RuntimeConfigurationError(
            f"{setting} must be http://127.0.0.1:{port}{path} for this environment"
        )
    return value.rstrip("/")


def load_gateway_runtime(environment: Mapping[str, str] | None = None) -> GatewayRuntime:
    values = os.environ if environment is None else environment
    deployment = _environment(values)
    host, port = _bind(values, role="gateway", deployment=deployment)

    base_path = _value(values, "NOVO_CHAT_BASE_PATH")
    if base_path != BASE_PATHS[deployment]:
        raise RuntimeConfigurationError(
            f"NOVO_CHAT_BASE_PATH must be {BASE_PATHS[deployment]} for {deployment}"
        )
    public_origin = _value(values, "NOVO_CHAT_PUBLIC_ORIGIN")
    origin = urlsplit(public_origin)
    try:
        origin.port
    except ValueError as exc:
        raise RuntimeConfigurationError("NOVO_CHAT_PUBLIC_ORIGIN contains an invalid port") from exc
    if (
        origin.scheme != "https"
        or not origin.netloc
        or not origin.hostname
        or origin.username
        or origin.password
        or origin.query
        or origin.fragment
        or origin.path not in {"", "/"}
    ):
        raise RuntimeConfigurationError("NOVO_CHAT_PUBLIC_ORIGIN must be a bare HTTPS origin")

    novo_url = _loopback_url(
        _value(values, "NOVO_INTEGRATION_API_URL"),
        setting="NOVO_INTEGRATION_API_URL",
        port=NOVO_PORTS[deployment],
        path="/api/integrations/v1",
    )
    worker_url = _loopback_url(
        _value(values, "NOVO_CHAT_WORKER_URL"),
        setting="NOVO_CHAT_WORKER_URL",
        port=REVERSE_FORWARD_PORTS[deployment],
        path="/internal/v1",
    )

    integration_file = _absolute_path(values, "NOVO_INTEGRATION_SECRET_FILE")
    csrf_file = _absolute_path(values, "NOVO_CHAT_CSRF_SECRET_FILE")
    request_file = _absolute_path(values, "NOVO_CHAT_WORKER_SIGNING_KEY_FILE")
    response_file = _absolute_path(values, "NOVO_CHAT_WORKER_RESPONSE_KEY_FILE")
    secret_paths = (integration_file, csrf_file, request_file, response_file)
    if len(set(secret_paths)) != len(secret_paths):
        raise RuntimeConfigurationError("gateway secrets must use four distinct files")
    secrets = (
        _read_secret(integration_file, setting="NOVO_INTEGRATION_SECRET_FILE"),
        _read_secret(csrf_file, setting="NOVO_CHAT_CSRF_SECRET_FILE"),
        _read_secret(request_file, setting="NOVO_CHAT_WORKER_SIGNING_KEY_FILE"),
        _read_secret(response_file, setting="NOVO_CHAT_WORKER_RESPONSE_KEY_FILE"),
    )
    if len(set(secrets)) != len(secrets):
        raise RuntimeConfigurationError("gateway secrets must use distinct values")
    try:
        secrets[0].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeConfigurationError("NOVO_INTEGRATION_SECRET_FILE must contain UTF-8 text") from exc

    request_key_id = _identifier(
        _value(values, "NOVO_CHAT_WORKER_REQUEST_KEY_ID"),
        setting="NOVO_CHAT_WORKER_REQUEST_KEY_ID",
        pattern=_KEY_ID,
    )
    response_key_id = _identifier(
        _value(values, "NOVO_CHAT_WORKER_RESPONSE_KEY_ID"),
        setting="NOVO_CHAT_WORKER_RESPONSE_KEY_ID",
        pattern=_KEY_ID,
    )
    if request_key_id == response_key_id:
        raise RuntimeConfigurationError("worker request and response key IDs must differ")

    job_db = _absolute_path(values, "NOVO_CHAT_JOB_DB")
    _state_target(job_db, setting="NOVO_CHAT_JOB_DB")
    try:
        settings = GatewaySettings(
            base_path=base_path,
            novo_api_base_url=novo_url,
            novo_integration_secret_file=integration_file,
            csrf_secret_file=csrf_file,
            novo_session_cookie_name=_value(values, "NOVO_SESSION_COOKIE_NAME"),
            novo_home_path=_value(values, "NOVO_HOME_PATH"),
            novo_login_path=_value(values, "NOVO_LOGIN_PATH"),
            novo_logout_path=_value(values, "NOVO_LOGOUT_PATH"),
            login_return_parameter=_value(values, "NOVO_LOGIN_RETURN_PARAMETER"),
            worker_base_url=worker_url,
            worker_signing_secret_file=request_file,
            worker_response_secret_file=response_file,
            worker_request_key_id=request_key_id,
            worker_response_key_id=response_key_id,
            environment=deployment,
            index_schema_version=_identifier(
                _value(values, "NOVO_CHAT_INDEX_SCHEMA_VERSION"),
                setting="NOVO_CHAT_INDEX_SCHEMA_VERSION",
                pattern=_SCHEMA_ID,
            ),
            job_db_path=job_db,
            job_ownership_ttl_s=int(
                _number(values, "NOVO_CHAT_JOB_OWNERSHIP_TTL_S", default="604800", integer=True)
            ),
            public_origin=public_origin,
            request_timeout_s=float(_number(values, "NOVO_CHAT_REQUEST_TIMEOUT_S", default="10")),
            gateway_max_request_bytes=int(
                _number(
                    values,
                    "NOVO_CHAT_GATEWAY_MAX_REQUEST_BYTES",
                    default=str(64 * 1024),
                    integer=True,
                )
            ),
            worker_job_timeout_s=float(_number(values, "NOVO_CHAT_WORKER_JOB_TIMEOUT_S", default="840")),
            worker_poll_interval_s=float(_number(values, "NOVO_CHAT_WORKER_POLL_INTERVAL_S", default="1")),
            novo_export_page_limit=int(_number(values, "NOVO_EXPORT_PAGE_LIMIT", default="8", integer=True)),
            ingest_batch_max_bytes=int(
                _number(
                    values,
                    "NOVO_CHAT_INGEST_BATCH_MAX_BYTES",
                    default=str(24 * 1024 * 1024),
                    integer=True,
                )
            ),
            novo_export_max_pages=int(_number(values, "NOVO_EXPORT_MAX_PAGES", default="100000", integer=True)),
            novo_export_max_batches=int(_number(values, "NOVO_EXPORT_MAX_BATCHES", default="10000", integer=True)),
            sync_revision_retries=_nonnegative_integer(
                values, "NOVO_CHAT_SYNC_REVISION_RETRIES", default="1"
            ),
        )
    except ValueError as exc:
        if isinstance(exc, RuntimeConfigurationError):
            raise
        raise RuntimeConfigurationError(str(exc)) from exc
    return GatewayRuntime(host=host, port=port, settings=settings)


def load_worker_runtime(
    environment: Mapping[str, str] | None = None,
    *,
    require_socket: bool = True,
) -> WorkerRuntime:
    values = os.environ if environment is None else environment
    deployment = _environment(values)
    host, port = _bind(values, role="worker", deployment=deployment)

    state_database = _absolute_path(values, "NOVO_CHAT_WORKER_STATE_DB")
    state_directory = _absolute_path(values, "NOVO_CHAT_WORKER_STATE_DIR")
    _state_target(state_database, setting="NOVO_CHAT_WORKER_STATE_DB")
    _state_target(state_directory, setting="NOVO_CHAT_WORKER_STATE_DIR", directory=True)

    request_key_id = _identifier(
        _value(values, "NOVO_CHAT_WORKER_REQUEST_KEY_ID"),
        setting="NOVO_CHAT_WORKER_REQUEST_KEY_ID",
        pattern=_KEY_ID,
    )
    response_key_id = _identifier(
        _value(values, "NOVO_CHAT_WORKER_RESPONSE_KEY_ID"),
        setting="NOVO_CHAT_WORKER_RESPONSE_KEY_ID",
        pattern=_KEY_ID,
    )
    if request_key_id == response_key_id:
        raise RuntimeConfigurationError("worker request and response key IDs must differ")
    request_file = _absolute_path(values, "NOVO_CHAT_WORKER_REQUEST_KEY_FILE")
    response_file = _absolute_path(values, "NOVO_CHAT_WORKER_RESPONSE_KEY_FILE")
    if request_file == response_file:
        raise RuntimeConfigurationError("worker request and response secrets must use distinct files")
    request_secret = _read_secret(request_file, setting="NOVO_CHAT_WORKER_REQUEST_KEY_FILE")
    response_secret = _read_secret(response_file, setting="NOVO_CHAT_WORKER_RESPONSE_KEY_FILE")
    if request_secret == response_secret:
        raise RuntimeConfigurationError("worker request and response secrets must use distinct values")

    models = _csv_identifiers(values, "NOVO_CHAT_APPROVED_MODELS", pattern=_MODEL_ID)
    schemas = _csv_identifiers(values, "NOVO_CHAT_INDEX_SCHEMA_VERSIONS", pattern=_SCHEMA_ID)
    backend_file = _absolute_path(values, "NOVO_CHAT_MODEL_BACKENDS_FILE")
    _readonly_config(backend_file, setting="NOVO_CHAT_MODEL_BACKENDS_FILE")
    try:
        backend_document = load_model_backend_document(backend_file)
        model_backend = HttpModelBackend(backend_document)
    except ValueError as exc:
        raise RuntimeConfigurationError("NOVO_CHAT_MODEL_BACKENDS_FILE is invalid") from exc
    if model_backend.approved_models != models:
        raise RuntimeConfigurationError(
            "NOVO_CHAT_APPROVED_MODELS must exactly match the ordered model backend IDs"
        )
    socket_path = _absolute_path(values, "NOVO_CHAT_MODELCTL_SOCKET")
    expected_socket = Path(f"/run/novo-chat/{deployment}-modelctl.sock")
    if socket_path != expected_socket:
        raise RuntimeConfigurationError(f"NOVO_CHAT_MODELCTL_SOCKET must be {expected_socket}")
    if require_socket:
        try:
            socket_metadata = socket_path.stat()
        except OSError as exc:
            raise RuntimeConfigurationError("NOVO_CHAT_MODELCTL_SOCKET is unavailable") from exc
        if not stat.S_ISSOCK(socket_metadata.st_mode):
            raise RuntimeConfigurationError("NOVO_CHAT_MODELCTL_SOCKET must name a Unix socket")
        if socket_metadata.st_uid != 0 or socket_metadata.st_mode & 0o007:
            raise RuntimeConfigurationError(
                "NOVO_CHAT_MODELCTL_SOCKET must be root-owned and inaccessible to other users"
            )
        if not os.access(socket_path, os.W_OK):
            raise RuntimeConfigurationError(
                "NOVO_CHAT_MODELCTL_SOCKET is not accessible to the worker socket group"
            )

    try:
        config = WorkerConfig(
            environment=deployment,
            state_database_path=state_database,
            state_directory=state_directory,
            request_secrets_by_key_id={request_key_id: request_secret},
            response_key_id=response_key_id,
            response_secret=response_secret,
            approved_models=models,
            index_schema_versions=schemas,
            executor_poll_seconds=float(
                _number(values, "NOVO_CHAT_EXECUTOR_POLL_SECONDS", default="0.5")
            ),
            model_start_timeout_seconds=float(
                _number(values, "NOVO_CHAT_MODEL_START_TIMEOUT_SECONDS", default="600")
            ),
            job_retention_seconds=int(
                _number(values, "NOVO_CHAT_JOB_RETENTION_SECONDS", default="604800", integer=True)
            ),
            orphan_grace_seconds=int(
                _number(values, "NOVO_CHAT_ORPHAN_GRACE_SECONDS", default="3600", integer=True)
            ),
            document_storage_quota_bytes=int(
                _number(
                    values,
                    "NOVO_CHAT_DOCUMENT_STORAGE_QUOTA_BYTES",
                    default=str(50 * 1024 * 1024 * 1024),
                    integer=True,
                )
            ),
            index_storage_quota_bytes=int(
                _number(
                    values,
                    "NOVO_CHAT_INDEX_STORAGE_QUOTA_BYTES",
                    default=str(50 * 1024 * 1024 * 1024),
                    integer=True,
                )
            ),
        )
    except ValueError as exc:
        if isinstance(exc, RuntimeConfigurationError):
            raise
        raise RuntimeConfigurationError(str(exc)) from exc
    return WorkerRuntime(
        host=host,
        port=port,
        config=config,
        controller=UnixSocketModelController(socket_path),
        model_backend=model_backend,
    )


def _serve_gateway() -> None:
    from .gateway.app import create_app

    runtime = load_gateway_runtime()
    app = create_app(runtime.settings)
    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
        server_header=False,
        date_header=False,
        timeout_graceful_shutdown=30,
    )


def _serve_worker() -> None:
    from .worker import create_worker_app

    runtime = load_worker_runtime()
    app = create_worker_app(
        runtime.config,
        model_controller=runtime.controller,
        compute_backend=runtime.model_backend,
        model_readiness_probe=runtime.model_backend,
    )
    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        timeout_graceful_shutdown=45,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novo-chat-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("gateway", help="validate configuration and serve the gateway")
    subparsers.add_parser("worker", help="validate configuration and serve the compute worker")
    check = subparsers.add_parser("check", help="validate configuration without opening a listener")
    check.add_argument("role", choices=("gateway", "worker"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "gateway":
            _serve_gateway()
        elif args.command == "worker":
            _serve_worker()
        elif args.role == "gateway":
            runtime = load_gateway_runtime()
            print(
                f"gateway configuration valid: environment={runtime.settings.environment} "
                f"bind={runtime.host}:{runtime.port}"
            )
        else:
            runtime = load_worker_runtime()
            print(
                f"worker configuration valid: environment={runtime.config.environment} "
                f"bind={runtime.host}:{runtime.port}"
            )
    except RuntimeConfigurationError as exc:
        print(f"novo-chat-runtime: configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_PATHS",
    "DEPLOYMENT_PORTS",
    "GatewayRuntime",
    "NOVO_PORTS",
    "REVERSE_FORWARD_PORTS",
    "RuntimeConfigurationError",
    "WorkerRuntime",
    "load_gateway_runtime",
    "load_worker_runtime",
    "main",
]
