"""Versioned Novo Chat gateway-to-worker protocol.

The HMAC covers an unambiguous canonical JSON document.  Callers must sign the
exact bytes they put on the wire; parsing and re-serializing JSON before signing
is deliberately not supported.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Mapping, Protocol, Sequence, TypeAlias
from urllib.parse import parse_qsl, quote, unquote, urlsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROTOCOL_VERSION = "1"
DEFAULT_AUDIENCE = "novo-chat-worker"
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 60
MAX_CITATION_EXCERPT_CHARS = 16_000

HEADER_PROTOCOL = "X-Novo-Chat-Protocol"
HEADER_AUDIENCE = "X-Novo-Chat-Audience"
HEADER_ENVIRONMENT = "X-Novo-Chat-Environment"
HEADER_KEY_ID = "X-Novo-Chat-Key-Id"
HEADER_TIMESTAMP = "X-Novo-Chat-Timestamp"
HEADER_NONCE = "X-Novo-Chat-Nonce"
HEADER_REQUEST_ID = "X-Novo-Chat-Request-Id"
HEADER_SIGNATURE = "X-Novo-Chat-Signature"
HEADER_STATUS = "X-Novo-Chat-Status"

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SAFE_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_ENVIRONMENT = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_HEX_SIGNATURE = re.compile(r"^[0-9a-f]{64}$")
_ENCODED_PATH_SEPARATOR = re.compile(r"%(?:2f|5c)", re.IGNORECASE)


class ProtocolError(Exception):
    """A safe, typed protocol rejection."""

    def __init__(self, code: str, message: str, status_code: int = 401, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ProtocolModel(BaseModel):
    """Strict JSON model with camelCase wire names and snake_case Python names."""

    model_config = ConfigDict(
        alias_generator=_camel,
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


def _validate_safe_token(value: str, label: str) -> str:
    if not _SAFE_TOKEN.fullmatch(value) or value == "*":
        raise ValueError(f"{label} is not a valid identifier")
    return value


def validate_same_origin_path(value: str) -> str:
    """Return a browser-safe same-origin absolute path.

    Backslashes and encoded separators are rejected because browser URL parsers
    can treat them as authority separators even when server-side parsers do not.
    """

    if (
        not value.startswith("/")
        or value.startswith("//")
        or "\\" in value
        or _ENCODED_PATH_SEPARATOR.search(value)
    ):
        raise ValueError("sourceUrl must be a same-origin absolute path")
    try:
        parsed = urlsplit(value)
        decoded = unquote(value, errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("sourceUrl must be a same-origin absolute path") from exc
    if parsed.scheme or parsed.netloc or "\\" in decoded or any(
        ord(character) < 32 or ord(character) == 127 for character in decoded
    ):
        raise ValueError("sourceUrl must be a same-origin absolute path")
    return value


def validate_request_id(value: str | UUID) -> str:
    """Return the canonical form of an unpredictable UUIDv4 request ID."""

    try:
        parsed = value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("request ID must be a UUIDv4") from exc
    if parsed.version != 4:
        raise ValueError("request ID must be a UUIDv4")
    return str(parsed)


def new_request_id() -> str:
    return str(uuid4())


class NotebookScope(ProtocolModel):
    notebook_id: str = Field(min_length=1, max_length=192)
    content_revision: str = Field(min_length=1, max_length=256)
    index_schema_version: str = Field(min_length=1, max_length=128)

    @field_validator("notebook_id")
    @classmethod
    def validate_notebook_id(cls, value: str) -> str:
        return _validate_safe_token(value, "notebookId")

    @field_validator("content_revision", "index_schema_version")
    @classmethod
    def validate_version_token(cls, value: str, info: Any) -> str:
        return _validate_safe_token(value, info.field_name)


class JobOperation(str, Enum):
    INGEST_BATCH = "ingest_batch"
    INGEST_FINALIZE = "ingest_finalize"
    INDEX_REBUILD = "index_rebuild"
    QUERY = "query"
    MODEL_START = "model_start"
    MODEL_STOP = "model_stop"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class QueryStrategy(str, Enum):
    HYBRID = "hybrid"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"


class WorkerState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"


class ModelRuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPING = "stopping"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class JobRequest(ProtocolModel):
    request_id: str
    idempotency_key: str = Field(min_length=8, max_length=192)
    actor_user_id: str = Field(min_length=1, max_length=192)
    scope: tuple[NotebookScope, ...] = Field(default_factory=tuple, max_length=256)

    @field_validator("request_id")
    @classmethod
    def validate_request_id_field(cls, value: str) -> str:
        return validate_request_id(value)

    @field_validator("idempotency_key", "actor_user_id")
    @classmethod
    def validate_request_token(cls, value: str, info: Any) -> str:
        return _validate_safe_token(value, info.field_name)

    @model_validator(mode="after")
    def unique_scope(self) -> "JobRequest":
        notebook_ids = [entry.notebook_id for entry in self.scope]
        if len(notebook_ids) != len(set(notebook_ids)):
            raise ValueError("scope contains a duplicate notebookId")
        return self


class IndexStatusRequest(ProtocolModel):
    request_id: str
    actor_user_id: str = Field(min_length=1, max_length=192)
    scope: tuple[NotebookScope, ...] = Field(min_length=1, max_length=256)

    @field_validator("request_id")
    @classmethod
    def validate_request_id_field(cls, value: str) -> str:
        return validate_request_id(value)

    @field_validator("actor_user_id")
    @classmethod
    def validate_actor(cls, value: str) -> str:
        return _validate_safe_token(value, "actorUserId")

    @model_validator(mode="after")
    def unique_scope(self) -> "IndexStatusRequest":
        notebook_ids = [entry.notebook_id for entry in self.scope]
        if len(notebook_ids) != len(set(notebook_ids)):
            raise ValueError("scope contains a duplicate notebookId")
        return self


class AttachmentMetadata(ProtocolModel):
    attachment_id: str = Field(min_length=1, max_length=192)
    name: str = Field(min_length=1, max_length=1024)
    mime_type: str = Field(default="application/octet-stream", max_length=255)
    size: int = Field(default=0, ge=0)
    block_type: str | None = Field(default=None, max_length=128)
    created_at: str | None = Field(default=None, max_length=64)

    @field_validator("attachment_id")
    @classmethod
    def validate_attachment_id(cls, value: str) -> str:
        return _validate_safe_token(value, "attachmentId")


class PageDocument(ProtocolModel):
    page_id: str = Field(min_length=1, max_length=192)
    title: str = Field(default="", max_length=4096)
    text: str = Field(default="", max_length=2_000_000)
    status: str | None = Field(default=None, max_length=128)
    created_at: str | None = Field(default=None, max_length=64)
    updated_at: str | None = Field(default=None, max_length=64)
    tags: tuple[str, ...] = Field(default_factory=tuple, max_length=256)
    attachments: tuple[AttachmentMetadata, ...] = Field(default_factory=tuple, max_length=1024)
    source_url: str = Field(min_length=1, max_length=2048)

    @field_validator("page_id")
    @classmethod
    def validate_page_id(cls, value: str) -> str:
        return _validate_safe_token(value, "pageId")

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return validate_same_origin_path(value)


class IngestBatchRequest(JobRequest):
    batch_number: int = Field(ge=0, le=1_000_000)
    batch_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    pages: tuple[PageDocument, ...] = Field(min_length=1, max_length=250)

    @model_validator(mode="after")
    def require_one_notebook(self) -> "IngestBatchRequest":
        if len(self.scope) != 1:
            raise ValueError("an ingest batch requires exactly one scope entry")
        return self


class IngestFinalizeRequest(JobRequest):
    document_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    page_count: int = Field(ge=0, le=10_000_000)
    batch_count: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def require_one_notebook(self) -> "IngestFinalizeRequest":
        if len(self.scope) != 1:
            raise ValueError("ingest finalization requires exactly one scope entry")
        return self


class RebuildRequest(JobRequest):
    force: bool = False

    @model_validator(mode="after")
    def require_scope(self) -> "RebuildRequest":
        if not self.scope:
            raise ValueError("an index rebuild requires at least one scope entry")
        return self


class QueryRequest(JobRequest):
    question: str = Field(min_length=1, max_length=32_000)
    retrieval_question: str | None = Field(default=None, min_length=1, max_length=32_000)
    model: str = Field(min_length=1, max_length=192)
    strategy: QueryStrategy = QueryStrategy.HYBRID
    max_sources: int = Field(default=16, ge=1, le=100)
    retrieval_top_k: int = Field(default=16, ge=1, le=32)

    @field_validator("model")
    @classmethod
    def validate_model_token(cls, value: str) -> str:
        return _validate_safe_token(value, "model")

    @model_validator(mode="after")
    def require_scope(self) -> "QueryRequest":
        if not self.scope:
            raise ValueError("a query requires at least one scope entry")
        return self


class ModelJobRequest(JobRequest):
    model: str = Field(min_length=1, max_length=192)

    @field_validator("model")
    @classmethod
    def validate_model_token(cls, value: str) -> str:
        return _validate_safe_token(value, "model")

    @model_validator(mode="after")
    def disallow_scope(self) -> "ModelJobRequest":
        if self.scope:
            raise ValueError("model lifecycle jobs do not accept notebook scope")
        return self


class IndexStatusItem(ProtocolModel):
    notebook_id: str
    content_revision: str
    index_schema_version: str
    exact_ready: bool
    activated_at: str | None = Field(default=None, max_length=64)
    chunk_count: int | None = Field(default=None, ge=0)


class IndexStatusResponse(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str
    indexes: tuple[IndexStatusItem, ...]


class HealthResponse(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str
    environment: str
    state: WorkerState
    queue_healthy: bool
    index_service_healthy: bool


class ModelDisplayDetails(ProtocolModel):
    """Optional, display-only metadata for one approved generation model."""

    model_size: str | None = Field(default=None, min_length=1, max_length=128, strict=True)
    max_tokens: int = Field(ge=1, le=8192, strict=True)
    max_model_len: int | None = Field(default=None, ge=1, le=2_000_000, strict=True)
    thinking: str | None = Field(default=None, min_length=1, max_length=128, strict=True)
    total_vram_gb: float | None = Field(
        default=None,
        gt=0,
        le=10_000,
        allow_inf_nan=False,
    )

    @field_validator("model_size", "thinking")
    @classmethod
    def validate_display_text(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("model display text must be a single printable line")
        return value

    @field_validator("total_vram_gb", mode="before")
    @classmethod
    def validate_total_vram_type(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("totalVramGb must be a JSON number")
        return value


class CapabilitiesResponse(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str
    operations: tuple[JobOperation, ...]
    approved_models: tuple[str, ...]
    index_schema_versions: tuple[str, ...]
    max_scope_entries: int
    max_ingest_pages_per_batch: int
    model_details: dict[str, ModelDisplayDetails] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model_details(self) -> "CapabilitiesResponse":
        approved = set(self.approved_models)
        for model_id in self.model_details:
            _validate_safe_token(model_id, "modelDetails key")
            if model_id not in approved:
                raise ValueError("modelDetails contains a model that is not approved")
        return self


class ModelStatusItem(ProtocolModel):
    model: str
    state: ModelRuntimeState


class ModelStatusResponse(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str
    models: tuple[ModelStatusItem, ...]


class Citation(ProtocolModel):
    notebook_id: str
    page_id: str
    source_url: str
    title: str = Field(default="", max_length=4096)
    excerpt: str = Field(default="", max_length=MAX_CITATION_EXCERPT_CHARS)
    score: float | None = Field(default=None, allow_inf_nan=False)
    source_idx: int | None = Field(default=None, ge=1, le=32)
    file: str | None = Field(default=None, min_length=1, max_length=1024)
    chunk_idx: int | None = Field(default=None, ge=0)
    bm25: float | None = Field(default=None, allow_inf_nan=False)
    dense: float | None = Field(default=None, allow_inf_nan=False)
    used_in_context: bool | None = None

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return validate_same_origin_path(value)


class QueryJobResult(ProtocolModel):
    kind: str = Field(default="query", pattern="^query$")
    answer: str
    model: str
    citations: tuple[Citation, ...]


class IndexJobResult(ProtocolModel):
    kind: str = Field(default="index", pattern="^index$")
    indexes: tuple[NotebookScope, ...]
    chunk_count: int = Field(default=0, ge=0)


class IngestJobResult(ProtocolModel):
    kind: str = Field(default="ingest", pattern="^ingest$")
    scope: NotebookScope
    accepted_pages: int = Field(default=0, ge=0)


class ModelJobResult(ProtocolModel):
    kind: str = Field(default="model", pattern="^model$")
    model: str
    state: ModelRuntimeState


JobResult: TypeAlias = QueryJobResult | IndexJobResult | IngestJobResult | ModelJobResult


class JobError(ProtocolModel):
    code: str
    message: str
    retryable: bool = False


class JobView(ProtocolModel):
    job_id: str
    operation: JobOperation
    state: JobState
    created_at: str
    updated_at: str
    progress: float = Field(ge=0.0, le=1.0)
    result: JobResult | None = None
    error: JobError | None = None


class JobSubmissionResponse(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str
    job: JobView


class JobStatusResponse(JobSubmissionResponse):
    pass


class ErrorResponse(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    request_id: str
    error: JobError


JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically as UTF-8 without insignificant spaces."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonicalizable JSON") from exc
    return rendered.encode("utf-8")


def body_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def ingest_pages_checksum(pages: Sequence[PageDocument | Mapping[str, Any]]) -> str:
    """Checksum the canonical, validated page-document representation."""

    normalized = [
        (page if isinstance(page, PageDocument) else PageDocument.model_validate(page)).model_dump(
            mode="json", by_alias=True
        )
        for page in pages
    ]
    return "sha256:" + body_sha256(canonical_json(normalized))


def document_pages_checksum(pages: Sequence[PageDocument | Mapping[str, Any]]) -> str:
    """Checksum a complete ordered notebook export using the ingest representation."""

    return ingest_pages_checksum(pages)


def canonicalize_path(path: str) -> str:
    if not path.startswith("/") or "?" in path or "#" in path or "\\" in path:
        raise ValueError("path must be an absolute URL path without query or fragment")
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("path contains invalid UTF-8 encoding") from exc
    if any(ord(character) < 32 for character in decoded):
        raise ValueError("path contains a control character")
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise ValueError("path contains a relative segment")
    return quote(decoded, safe="/-._~")


def canonicalize_query(query: str | bytes | Mapping[str, Any] | Sequence[tuple[str, Any]] = "") -> str:
    if isinstance(query, bytes):
        query = query.decode("ascii", errors="strict")
    if isinstance(query, str):
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=False, encoding="utf-8", errors="strict")
    elif isinstance(query, Mapping):
        pairs = []
        for key, value in query.items():
            if isinstance(value, (list, tuple)):
                pairs.extend((str(key), str(item)) for item in value)
            else:
                pairs.append((str(key), str(value)))
    else:
        pairs = [(str(key), str(value)) for key, value in query]
    encoded = [(quote(key, safe="-._~"), quote(value, safe="-._~")) for key, value in pairs]
    encoded.sort()
    return "&".join(f"{key}={value}" for key, value in encoded)


def _secret_bytes(secret: bytes | str) -> bytes:
    value = secret.encode("utf-8") if isinstance(secret, str) else secret
    if not isinstance(value, bytes) or len(value) < 32:
        raise ValueError("HMAC secrets must contain at least 32 bytes")
    return value


def _validate_signing_metadata(key_id: str, environment: str, audience: str, request_id: str) -> None:
    if not _SAFE_KEY_ID.fullmatch(key_id):
        raise ValueError("invalid key ID")
    if not _SAFE_ENVIRONMENT.fullmatch(environment):
        raise ValueError("invalid environment")
    if not _SAFE_TOKEN.fullmatch(audience):
        raise ValueError("invalid audience")
    validate_request_id(request_id)


def _request_document(
    *,
    protocol_version: str,
    audience: str,
    environment: str,
    key_id: str,
    timestamp: int,
    nonce: str,
    request_id: str,
    method: str,
    path: str,
    query: str | bytes | Mapping[str, Any] | Sequence[tuple[str, Any]],
    body: bytes,
) -> dict[str, Any]:
    return {
        "protocolVersion": protocol_version,
        "audience": audience,
        "environment": environment,
        "keyId": key_id,
        "timestamp": timestamp,
        "nonce": nonce,
        "requestId": request_id,
        "method": method.upper(),
        "path": canonicalize_path(path),
        "query": canonicalize_query(query),
        "bodySha256": body_sha256(body),
    }


def sign_request(
    *,
    secret: bytes | str,
    key_id: str,
    environment: str,
    audience: str = DEFAULT_AUDIENCE,
    request_id: str,
    method: str,
    path: str,
    query: str | bytes | Mapping[str, Any] | Sequence[tuple[str, Any]] = "",
    body: bytes = b"",
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Return the complete HMAC request-header set for exact wire bytes."""

    _validate_signing_metadata(key_id, environment, audience, request_id)
    actual_timestamp = int(time.time()) if timestamp is None else int(timestamp)
    actual_nonce = secrets.token_hex(24) if nonce is None else nonce
    if not _SAFE_TOKEN.fullmatch(actual_nonce):
        raise ValueError("invalid nonce")
    document = _request_document(
        protocol_version=PROTOCOL_VERSION,
        audience=audience,
        environment=environment,
        key_id=key_id,
        timestamp=actual_timestamp,
        nonce=actual_nonce,
        request_id=request_id,
        method=method,
        path=path,
        query=query,
        body=body,
    )
    signature = hmac.new(_secret_bytes(secret), canonical_json(document), hashlib.sha256).hexdigest()
    return {
        HEADER_PROTOCOL: PROTOCOL_VERSION,
        HEADER_AUDIENCE: audience,
        HEADER_ENVIRONMENT: environment,
        HEADER_KEY_ID: key_id,
        HEADER_TIMESTAMP: str(actual_timestamp),
        HEADER_NONCE: actual_nonce,
        HEADER_REQUEST_ID: validate_request_id(request_id),
        HEADER_SIGNATURE: signature,
    }


class ReplayGuard(Protocol):
    def claim(self, *, environment: str, key_id: str, nonce: str, seen_at: int, expires_at: int) -> bool:
        """Atomically return True only for a nonce not previously claimed."""


class InMemoryReplayGuard:
    """Process-local replay guard useful for clients and tests."""

    def __init__(self) -> None:
        self._claims: dict[tuple[str, str, str], int] = {}
        self._lock = Lock()

    def claim(self, *, environment: str, key_id: str, nonce: str, seen_at: int, expires_at: int) -> bool:
        claim_key = (environment, key_id, nonce)
        with self._lock:
            expired = [key for key, expiry in self._claims.items() if expiry < seen_at]
            for key in expired:
                self._claims.pop(key, None)
            if claim_key in self._claims:
                return False
            self._claims[claim_key] = expires_at
            return True


@dataclass(frozen=True)
class VerifiedRequest:
    protocol_version: str
    audience: str
    environment: str
    key_id: str
    timestamp: int
    nonce: str
    request_id: str


def _header_map(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _required_header(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name.lower())
    if not value:
        raise ProtocolError("AUTHENTICATION_REQUIRED", "Request authentication failed.")
    return value


def verify_request(
    *,
    headers: Mapping[str, str],
    secrets_by_key_id: Mapping[str, bytes | str],
    expected_environment: str,
    expected_audience: str = DEFAULT_AUDIENCE,
    method: str,
    path: str,
    query: str | bytes | Mapping[str, Any] | Sequence[tuple[str, Any]] = "",
    body: bytes = b"",
    replay_guard: ReplayGuard | None = None,
    now: int | None = None,
    max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
) -> VerifiedRequest:
    normalized = _header_map(headers)
    protocol_version = _required_header(normalized, HEADER_PROTOCOL)
    audience = _required_header(normalized, HEADER_AUDIENCE)
    environment = _required_header(normalized, HEADER_ENVIRONMENT)
    key_id = _required_header(normalized, HEADER_KEY_ID)
    timestamp_text = _required_header(normalized, HEADER_TIMESTAMP)
    nonce = _required_header(normalized, HEADER_NONCE)
    request_id = _required_header(normalized, HEADER_REQUEST_ID)
    supplied_signature = _required_header(normalized, HEADER_SIGNATURE)

    if protocol_version != PROTOCOL_VERSION:
        raise ProtocolError("UNSUPPORTED_PROTOCOL", "Request protocol version is not supported.", 400)
    if audience != expected_audience or environment != expected_environment:
        raise ProtocolError("AUTHENTICATION_FAILED", "Request authentication failed.")
    if not _SAFE_KEY_ID.fullmatch(key_id) or not _SAFE_TOKEN.fullmatch(nonce):
        raise ProtocolError("AUTHENTICATION_FAILED", "Request authentication failed.")
    try:
        request_id = validate_request_id(request_id)
        timestamp = int(timestamp_text)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("AUTHENTICATION_FAILED", "Request authentication failed.") from exc
    if not _HEX_SIGNATURE.fullmatch(supplied_signature):
        raise ProtocolError("AUTHENTICATION_FAILED", "Request authentication failed.")
    secret = secrets_by_key_id.get(key_id)
    if secret is None:
        raise ProtocolError("AUTHENTICATION_FAILED", "Request authentication failed.")
    actual_now = int(time.time()) if now is None else int(now)
    if abs(actual_now - timestamp) > max_clock_skew_seconds:
        raise ProtocolError("REQUEST_EXPIRED", "Signed request is outside the allowed time window.")

    try:
        document = _request_document(
            protocol_version=protocol_version,
            audience=audience,
            environment=environment,
            key_id=key_id,
            timestamp=timestamp,
            nonce=nonce,
            request_id=request_id,
            method=method,
            path=path,
            query=query,
            body=body,
        )
        expected_signature = hmac.new(_secret_bytes(secret), canonical_json(document), hashlib.sha256).hexdigest()
    except ValueError as exc:
        raise ProtocolError("AUTHENTICATION_FAILED", "Request authentication failed.") from exc
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise ProtocolError("AUTHENTICATION_FAILED", "Request authentication failed.")
    if replay_guard is not None and not replay_guard.claim(
        environment=environment,
        key_id=key_id,
        nonce=nonce,
        seen_at=actual_now,
        expires_at=timestamp + max_clock_skew_seconds,
    ):
        raise ProtocolError("REQUEST_REPLAYED", "Signed request nonce has already been used.", 409)
    return VerifiedRequest(
        protocol_version=protocol_version,
        audience=audience,
        environment=environment,
        key_id=key_id,
        timestamp=timestamp,
        nonce=nonce,
        request_id=request_id,
    )


def _response_document(
    *,
    protocol_version: str,
    audience: str,
    environment: str,
    key_id: str,
    timestamp: int,
    request_id: str,
    status_code: int,
    body: bytes,
) -> dict[str, Any]:
    return {
        "protocolVersion": protocol_version,
        "audience": audience,
        "environment": environment,
        "keyId": key_id,
        "timestamp": timestamp,
        "requestId": request_id,
        "status": status_code,
        "bodySha256": body_sha256(body),
    }


def sign_response(
    *,
    secret: bytes | str,
    key_id: str,
    environment: str,
    request_id: str,
    status_code: int,
    body: bytes,
    audience: str = DEFAULT_AUDIENCE,
    timestamp: int | None = None,
) -> dict[str, str]:
    """Return response headers authenticated to a request ID and HTTP status."""

    _validate_signing_metadata(key_id, environment, audience, request_id)
    actual_timestamp = int(time.time()) if timestamp is None else int(timestamp)
    document = _response_document(
        protocol_version=PROTOCOL_VERSION,
        audience=audience,
        environment=environment,
        key_id=key_id,
        timestamp=actual_timestamp,
        request_id=request_id,
        status_code=int(status_code),
        body=body,
    )
    signature = hmac.new(_secret_bytes(secret), canonical_json(document), hashlib.sha256).hexdigest()
    return {
        HEADER_PROTOCOL: PROTOCOL_VERSION,
        HEADER_AUDIENCE: audience,
        HEADER_ENVIRONMENT: environment,
        HEADER_KEY_ID: key_id,
        HEADER_TIMESTAMP: str(actual_timestamp),
        HEADER_REQUEST_ID: validate_request_id(request_id),
        HEADER_STATUS: str(int(status_code)),
        HEADER_SIGNATURE: signature,
    }


@dataclass(frozen=True)
class VerifiedResponse:
    protocol_version: str
    audience: str
    environment: str
    key_id: str
    timestamp: int
    request_id: str
    status_code: int


def verify_response(
    *,
    headers: Mapping[str, str],
    secrets_by_key_id: Mapping[str, bytes | str],
    expected_environment: str,
    request_id: str,
    status_code: int,
    body: bytes,
    expected_audience: str = DEFAULT_AUDIENCE,
    now: int | None = None,
    max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
) -> VerifiedResponse:
    normalized = _header_map(headers)
    protocol_version = _required_header(normalized, HEADER_PROTOCOL)
    audience = _required_header(normalized, HEADER_AUDIENCE)
    environment = _required_header(normalized, HEADER_ENVIRONMENT)
    key_id = _required_header(normalized, HEADER_KEY_ID)
    timestamp_text = _required_header(normalized, HEADER_TIMESTAMP)
    signed_request_id = _required_header(normalized, HEADER_REQUEST_ID)
    status_text = _required_header(normalized, HEADER_STATUS)
    supplied_signature = _required_header(normalized, HEADER_SIGNATURE)
    if protocol_version != PROTOCOL_VERSION:
        raise ProtocolError("UNSUPPORTED_PROTOCOL", "Response protocol version is not supported.", 502)
    if audience != expected_audience or environment != expected_environment:
        raise ProtocolError("RESPONSE_AUTHENTICATION_FAILED", "Worker response authentication failed.", 502)
    try:
        signed_request_id = validate_request_id(signed_request_id)
        expected_request_id = validate_request_id(request_id)
        timestamp = int(timestamp_text)
        signed_status = int(status_text)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("RESPONSE_AUTHENTICATION_FAILED", "Worker response authentication failed.", 502) from exc
    if signed_request_id != expected_request_id or signed_status != int(status_code):
        raise ProtocolError("RESPONSE_AUTHENTICATION_FAILED", "Worker response authentication failed.", 502)
    if not _HEX_SIGNATURE.fullmatch(supplied_signature):
        raise ProtocolError("RESPONSE_AUTHENTICATION_FAILED", "Worker response authentication failed.", 502)
    actual_now = int(time.time()) if now is None else int(now)
    if abs(actual_now - timestamp) > max_clock_skew_seconds:
        raise ProtocolError("RESPONSE_EXPIRED", "Worker response is outside the allowed time window.", 502)
    secret = secrets_by_key_id.get(key_id)
    if secret is None:
        raise ProtocolError("RESPONSE_AUTHENTICATION_FAILED", "Worker response authentication failed.", 502)
    try:
        document = _response_document(
            protocol_version=protocol_version,
            audience=audience,
            environment=environment,
            key_id=key_id,
            timestamp=timestamp,
            request_id=signed_request_id,
            status_code=signed_status,
            body=body,
        )
        expected_signature = hmac.new(_secret_bytes(secret), canonical_json(document), hashlib.sha256).hexdigest()
    except ValueError as exc:
        raise ProtocolError("RESPONSE_AUTHENTICATION_FAILED", "Worker response authentication failed.", 502) from exc
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise ProtocolError("RESPONSE_AUTHENTICATION_FAILED", "Worker response authentication failed.", 502)
    return VerifiedResponse(
        protocol_version=protocol_version,
        audience=audience,
        environment=environment,
        key_id=key_id,
        timestamp=timestamp,
        request_id=signed_request_id,
        status_code=signed_status,
    )


__all__ = [
    "PROTOCOL_VERSION",
    "DEFAULT_AUDIENCE",
    "AttachmentMetadata",
    "CapabilitiesResponse",
    "Citation",
    "ErrorResponse",
    "HealthResponse",
    "IndexJobResult",
    "IndexStatusItem",
    "IndexStatusRequest",
    "IndexStatusResponse",
    "IngestBatchRequest",
    "IngestFinalizeRequest",
    "IngestJobResult",
    "InMemoryReplayGuard",
    "JobError",
    "JobOperation",
    "JobRequest",
    "JobState",
    "JobStatusResponse",
    "JobSubmissionResponse",
    "JobView",
    "MAX_CITATION_EXCERPT_CHARS",
    "ModelDisplayDetails",
    "ModelJobRequest",
    "ModelJobResult",
    "ModelRuntimeState",
    "ModelStatusItem",
    "ModelStatusResponse",
    "NotebookScope",
    "PageDocument",
    "ProtocolError",
    "QueryJobResult",
    "QueryRequest",
    "QueryStrategy",
    "RebuildRequest",
    "VerifiedRequest",
    "VerifiedResponse",
    "WorkerState",
    "body_sha256",
    "canonical_json",
    "canonicalize_path",
    "canonicalize_query",
    "document_pages_checksum",
    "ingest_pages_checksum",
    "new_request_id",
    "sign_request",
    "sign_response",
    "validate_request_id",
    "verify_request",
    "verify_response",
    "validate_same_origin_path",
]
