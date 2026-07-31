"""Configurable loopback-only embedding, generation, and readiness backend."""

from __future__ import annotations

import ipaddress
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .compute import ComputeError
from .protocol import ModelDisplayDetails, NotebookScope
from .rag_core import generation_messages


_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _loopback_origin(value: str) -> str:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("backend URL contains an invalid port") from exc
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or port is None
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("backend URL must be a plain HTTP loopback origin")
    host = parsed.hostname.lower().rstrip(".")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("backend URL must use an explicit loopback host")
        except ValueError as exc:
            raise ValueError("backend URL must use an explicit loopback host") from exc
    return value.rstrip("/")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EmbeddingBackendConfig(_StrictModel):
    base_url: str = Field(alias="baseUrl")
    model: str = Field(min_length=1, max_length=192)
    max_characters: int = Field(default=2400, alias="maxCharacters", ge=128, le=100_000)
    document_keep_alive: str = Field(default="10m", alias="documentKeepAlive", max_length=32)
    query_keep_alive: str = Field(default="0", alias="queryKeepAlive", max_length=32)
    timeout_seconds: float = Field(default=120.0, alias="timeoutSeconds", ge=1, le=600)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _loopback_origin(value)


class GenerationBackendConfig(_StrictModel):
    base_url: str = Field(alias="baseUrl")
    served_model: str = Field(alias="servedModel", min_length=1, max_length=256)
    model_size: str | None = Field(
        default=None,
        alias="modelSize",
        min_length=1,
        max_length=128,
        strict=True,
    )
    max_tokens: int = Field(default=2048, alias="maxTokens", ge=1, le=8192)
    max_model_len: int | None = Field(
        default=None,
        alias="maxModelLen",
        ge=1,
        le=2_000_000,
        strict=True,
    )
    thinking: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        strict=True,
    )
    total_vram_gb: float | None = Field(
        default=None,
        alias="totalVramGb",
        gt=0,
        le=10_000,
        allow_inf_nan=False,
    )
    timeout_seconds: float = Field(default=840.0, alias="timeoutSeconds", ge=1, le=900)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict, alias="chatTemplateKwargs")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _loopback_origin(value)

    @field_validator("model_size", "thinking")
    @classmethod
    def validate_display_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or any(
            ord(character) < 32 or ord(character) == 127 for character in normalized
        ):
            raise ValueError("model display text must be a single printable line")
        return normalized

    @field_validator("total_vram_gb", mode="before")
    @classmethod
    def validate_total_vram_type(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("totalVramGb must be a JSON number")
        return value

    @field_validator("chat_template_kwargs")
    @classmethod
    def validate_template_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 8192:
            raise ValueError("chatTemplateKwargs is too large")
        return value


class ModelBackendDocument(_StrictModel):
    version: int
    embedding: EmbeddingBackendConfig
    models: dict[str, GenerationBackendConfig]

    @model_validator(mode="after")
    def validate_document(self) -> "ModelBackendDocument":
        if self.version != 1:
            raise ValueError("model backend configuration must use version 1")
        if not self.models:
            raise ValueError("model backend configuration must define at least one model")
        if any(not _MODEL_ID.fullmatch(model_id) for model_id in self.models):
            raise ValueError("model backend configuration contains an invalid model ID")
        return self


def load_model_backend_document(path: str | Path) -> ModelBackendDocument:
    config_path = Path(path)
    if not config_path.is_absolute():
        raise ValueError("model backend configuration path must be absolute")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model backend configuration is unavailable or invalid") from exc
    return ModelBackendDocument.model_validate(document)


class HttpModelBackend:
    """Synchronous adapter used only by the worker's serialized executor."""

    def __init__(self, config: ModelBackendDocument, *, session: requests.Session | None = None):
        self.config = config
        self._session = session or requests.Session()
        self._session.trust_env = False

    @property
    def approved_models(self) -> tuple[str, ...]:
        return tuple(self.config.models)

    @property
    def model_details(self) -> dict[str, ModelDisplayDetails]:
        return {
            model_id: ModelDisplayDetails(
                model_size=backend.model_size,
                max_tokens=backend.max_tokens,
                max_model_len=backend.max_model_len,
                thinking=backend.thinking,
                total_vram_gb=backend.total_vram_gb,
            )
            for model_id, backend in self.config.models.items()
        }

    def embed_documents(self, texts: Sequence[str], *, scope: NotebookScope) -> np.ndarray:
        del scope
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([self._embed(text, kind="document") for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text, kind="query")

    def _embed(self, text: str, *, kind: str) -> np.ndarray:
        embedding = self.config.embedding
        prefix = "search_query: " if kind == "query" else "search_document: "
        payload = {
            "model": embedding.model,
            "prompt": prefix + text[: embedding.max_characters],
            "keep_alive": (
                embedding.query_keep_alive if kind == "query" else embedding.document_keep_alive
            ),
        }
        for attempt in range(4):
            try:
                response = self._session.post(
                    f"{embedding.base_url}/api/embeddings",
                    json=payload,
                    timeout=embedding.timeout_seconds,
                )
                response.raise_for_status()
                vector = np.asarray(response.json()["embedding"], dtype=np.float32).reshape(-1)
                if vector.size < 1 or not np.all(np.isfinite(vector)):
                    raise ValueError("invalid embedding")
                return vector
            except (requests.RequestException, KeyError, TypeError, ValueError):
                if attempt < 3:
                    time.sleep(2 + attempt * 4)
        raise ComputeError(
            "EMBEDDING_UNAVAILABLE",
            "The configured embedding service is unavailable.",
            retryable=True,
        )

    def generate(
        self,
        question: str,
        hits: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_sources: int,
    ) -> str:
        if not hits:
            return "The provided documents do not contain this information."
        backend = self.config.models.get(model)
        if backend is None:
            raise ComputeError("MODEL_NOT_APPROVED", "Requested model is not available.")
        payload: dict[str, Any] = {
            "model": backend.served_model,
            "messages": generation_messages(question, hits, max_sources=max_sources),
            "max_tokens": backend.max_tokens,
            "temperature": 0.2,
            "stream": False,
        }
        if backend.chat_template_kwargs:
            payload["chat_template_kwargs"] = backend.chat_template_kwargs
        try:
            response = self._session.post(
                f"{backend.base_url}/v1/chat/completions",
                json=payload,
                timeout=backend.timeout_seconds,
            )
            response.raise_for_status()
            choice = (response.json().get("choices") or [{}])[0]
            answer = str((choice.get("message") or {}).get("content") or "").strip()
        except (requests.RequestException, AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ComputeError(
                "GENERATION_UNAVAILABLE",
                "The configured generation service is unavailable.",
                retryable=True,
            ) from exc
        if not answer:
            raise ComputeError(
                "GENERATION_EMPTY",
                "The configured generation service returned no answer.",
                retryable=True,
            )
        return answer

    def is_ready(self, model: str) -> bool:
        backend = self.config.models.get(model)
        if backend is None:
            return False
        try:
            response = self._session.get(f"{backend.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(self, model: str, *, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while time.monotonic() < deadline:
            if self.is_ready(model):
                return True
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        return self.is_ready(model)


class UnavailableModelBackend:
    """Fail-closed default for tests or an intentionally unconfigured worker."""

    def embed_documents(self, texts: Sequence[str], *, scope: NotebookScope) -> np.ndarray:
        del scope
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        raise ComputeError("MODEL_BACKEND_UNAVAILABLE", "Model backend is unavailable.", retryable=True)

    def embed_query(self, text: str) -> np.ndarray:
        del text
        raise ComputeError("MODEL_BACKEND_UNAVAILABLE", "Model backend is unavailable.", retryable=True)

    def generate(
        self,
        question: str,
        hits: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_sources: int,
    ) -> str:
        del question, hits, model, max_sources
        raise ComputeError("MODEL_BACKEND_UNAVAILABLE", "Model backend is unavailable.", retryable=True)

    def is_ready(self, model: str) -> bool:
        del model
        return False

    def wait_until_ready(self, model: str, *, timeout_seconds: float) -> bool:
        del model, timeout_seconds
        return False


__all__ = [
    "HttpModelBackend",
    "ModelBackendDocument",
    "UnavailableModelBackend",
    "load_model_backend_document",
]
