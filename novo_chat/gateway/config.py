from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


_COOKIE_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENVIRONMENT = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_PROTOCOL_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _base_path(value: str) -> str:
    value = value.strip()
    if not value.startswith("/"):
        value = f"/{value}"
    value = value.rstrip("/")
    if not value or value == "/":
        raise ValueError("NOVO_CHAT_BASE_PATH must name a non-root URL path")
    if "//" in value or "?" in value or "#" in value:
        raise ValueError("NOVO_CHAT_BASE_PATH must be a simple absolute URL path")
    return value


def _same_origin_path(value: str, setting: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or not value.startswith("/") or value.startswith("//"):
        raise ValueError(f"{setting} must be a same-origin absolute path")
    return value


def _loopback_url(value: str, setting: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{setting} must be an HTTP(S) URL without user information")
    host = parsed.hostname.lower().rstrip(".")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"{setting} must resolve through an explicit loopback host") from exc
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    """Configuration for the public gateway.

    Both upstreams are deliberately constrained to literal loopback addresses.
    That makes it impossible for a configuration typo to leak ``eln_session``
    to a remote Novo endpoint or expose the worker transport directly.
    """

    base_path: str = "/chat"
    novo_api_base_url: str = "http://127.0.0.1:3148/api/integrations/v1"
    novo_integration_secret_file: Path = Path("/run/secrets/novo-chat-integration-token")
    csrf_secret_file: Path | None = None
    novo_session_cookie_name: str = "eln_session"
    novo_home_path: str = "/"
    novo_login_path: str = "/"
    novo_logout_path: str = "/api/auth/logout"
    login_return_parameter: str = "returnTo"
    worker_base_url: str = "http://127.0.0.1:18095/internal/v1"
    worker_signing_secret_file: Path = Path("/run/secrets/novo-chat-worker-signing-key")
    worker_response_secret_file: Path | None = None
    worker_request_key_id: str = "gateway"
    worker_response_key_id: str = "compute-worker"
    environment: str = "development"
    index_schema_version: str = "novo-chat-index-v1"
    job_db_path: Path = Path("/var/lib/novo-chat-gateway/jobs.sqlite3")
    job_ownership_ttl_s: int = 7 * 24 * 60 * 60
    public_origin: str | None = None
    request_timeout_s: float = 10.0
    gateway_max_request_bytes: int = 64 * 1024
    worker_job_timeout_s: float = 840.0
    worker_poll_interval_s: float = 1.0
    novo_export_page_limit: int = 8
    ingest_batch_max_bytes: int = 24 * 1024 * 1024
    novo_export_max_pages: int = 100_000
    novo_export_max_batches: int = 10_000
    sync_revision_retries: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_path", _base_path(self.base_path))
        object.__setattr__(
            self,
            "novo_api_base_url",
            _loopback_url(self.novo_api_base_url, "NOVO_INTEGRATION_API_URL"),
        )
        object.__setattr__(self, "worker_base_url", _loopback_url(self.worker_base_url, "NOVO_CHAT_WORKER_URL"))
        object.__setattr__(self, "novo_home_path", _same_origin_path(self.novo_home_path, "NOVO_HOME_PATH"))
        object.__setattr__(self, "novo_login_path", _same_origin_path(self.novo_login_path, "NOVO_LOGIN_PATH"))
        object.__setattr__(self, "novo_logout_path", _same_origin_path(self.novo_logout_path, "NOVO_LOGOUT_PATH"))
        if self.public_origin:
            parsed = urlsplit(self.public_origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
                raise ValueError("NOVO_CHAT_PUBLIC_ORIGIN must contain only an HTTP(S) origin")
            object.__setattr__(self, "public_origin", self.public_origin.rstrip("/"))
        if not _COOKIE_NAME.fullmatch(self.novo_session_cookie_name):
            raise ValueError("NOVO_SESSION_COOKIE_NAME must be a valid cookie token")
        if not _KEY_ID.fullmatch(self.worker_request_key_id) or not _KEY_ID.fullmatch(self.worker_response_key_id):
            raise ValueError("worker signing key IDs must be valid protocol identifiers")
        if not _ENVIRONMENT.fullmatch(self.environment):
            raise ValueError("NOVO_CHAT_ENVIRONMENT must be a lowercase protocol environment")
        if not _PROTOCOL_TOKEN.fullmatch(self.index_schema_version):
            raise ValueError("NOVO_CHAT_INDEX_SCHEMA_VERSION must be a valid protocol token")
        if self.request_timeout_s <= 0 or self.worker_job_timeout_s <= 0 or self.worker_poll_interval_s <= 0:
            raise ValueError("gateway timeouts must be positive")
        if not 1024 <= self.gateway_max_request_bytes <= 1024 * 1024:
            raise ValueError("NOVO_CHAT_GATEWAY_MAX_REQUEST_BYTES must be between 1 KiB and 1 MiB")
        if self.job_ownership_ttl_s < self.worker_job_timeout_s:
            raise ValueError("NOVO_CHAT_JOB_OWNERSHIP_TTL_S must cover the worker job timeout")
        if not 1 <= self.novo_export_page_limit <= 250:
            raise ValueError("NOVO_EXPORT_PAGE_LIMIT must be between 1 and 250")
        if not 1 <= self.ingest_batch_max_bytes <= 30 * 1024 * 1024:
            raise ValueError("NOVO_CHAT_INGEST_BATCH_MAX_BYTES must be between 1 byte and 30 MiB")
        if self.novo_export_max_pages < self.novo_export_page_limit or self.novo_export_max_batches < 1:
            raise ValueError("Novo export caps are invalid")
        if not 0 <= self.sync_revision_retries <= 3:
            raise ValueError("NOVO_CHAT_SYNC_REVISION_RETRIES must be between 0 and 3")

    @property
    def effective_csrf_secret_file(self) -> Path:
        return self.csrf_secret_file or self.novo_integration_secret_file

    @property
    def effective_worker_response_secret_file(self) -> Path:
        return self.worker_response_secret_file or self.worker_signing_secret_file

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        csrf_file = os.getenv("NOVO_CHAT_CSRF_SECRET_FILE")
        response_secret_file = os.getenv("NOVO_CHAT_WORKER_RESPONSE_KEY_FILE")
        public_origin = os.getenv("NOVO_CHAT_PUBLIC_ORIGIN")
        return cls(
            base_path=os.getenv("NOVO_CHAT_BASE_PATH", "/chat"),
            novo_api_base_url=os.getenv(
                "NOVO_INTEGRATION_API_URL", "http://127.0.0.1:3148/api/integrations/v1"
            ),
            novo_integration_secret_file=Path(
                os.getenv("NOVO_INTEGRATION_SECRET_FILE", "/run/secrets/novo-chat-integration-token")
            ),
            csrf_secret_file=Path(csrf_file) if csrf_file else None,
            novo_session_cookie_name=os.getenv("NOVO_SESSION_COOKIE_NAME", "eln_session"),
            novo_home_path=os.getenv("NOVO_HOME_PATH", "/"),
            novo_login_path=os.getenv("NOVO_LOGIN_PATH", "/"),
            novo_logout_path=os.getenv("NOVO_LOGOUT_PATH", "/api/auth/logout"),
            login_return_parameter=os.getenv("NOVO_LOGIN_RETURN_PARAMETER", "returnTo"),
            worker_base_url=os.getenv("NOVO_CHAT_WORKER_URL", "http://127.0.0.1:18095/internal/v1"),
            worker_signing_secret_file=Path(
                os.getenv("NOVO_CHAT_WORKER_SIGNING_KEY_FILE", "/run/secrets/novo-chat-worker-signing-key")
            ),
            worker_response_secret_file=Path(response_secret_file) if response_secret_file else None,
            worker_request_key_id=os.getenv("NOVO_CHAT_WORKER_REQUEST_KEY_ID", "gateway"),
            worker_response_key_id=os.getenv("NOVO_CHAT_WORKER_RESPONSE_KEY_ID", "compute-worker"),
            environment=os.getenv("NOVO_CHAT_ENVIRONMENT", "development"),
            index_schema_version=os.getenv("NOVO_CHAT_INDEX_SCHEMA_VERSION", "novo-chat-index-v1"),
            job_db_path=Path(os.getenv("NOVO_CHAT_JOB_DB", "/var/lib/novo-chat-gateway/jobs.sqlite3")),
            job_ownership_ttl_s=int(os.getenv("NOVO_CHAT_JOB_OWNERSHIP_TTL_S", "604800")),
            public_origin=public_origin,
            request_timeout_s=float(os.getenv("NOVO_CHAT_REQUEST_TIMEOUT_S", "10")),
            gateway_max_request_bytes=int(os.getenv("NOVO_CHAT_GATEWAY_MAX_REQUEST_BYTES", str(64 * 1024))),
            worker_job_timeout_s=float(os.getenv("NOVO_CHAT_WORKER_JOB_TIMEOUT_S", "840")),
            worker_poll_interval_s=float(os.getenv("NOVO_CHAT_WORKER_POLL_INTERVAL_S", "1")),
            novo_export_page_limit=int(os.getenv("NOVO_EXPORT_PAGE_LIMIT", "8")),
            ingest_batch_max_bytes=int(
                os.getenv("NOVO_CHAT_INGEST_BATCH_MAX_BYTES", str(24 * 1024 * 1024))
            ),
            novo_export_max_pages=int(os.getenv("NOVO_EXPORT_MAX_PAGES", "100000")),
            novo_export_max_batches=int(os.getenv("NOVO_EXPORT_MAX_BATCHES", "10000")),
            sync_revision_retries=int(os.getenv("NOVO_CHAT_SYNC_REVISION_RETRIES", "1")),
        )
