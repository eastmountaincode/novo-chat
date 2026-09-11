"""Configurable loopback-only embedding, generation, and readiness backend."""

from __future__ import annotations

import ipaddress
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .compute import ComputeError
from .protocol import (
    GenerationResult,
    ModelDisplayDetails,
    NotebookScope,
    QueryTimings,
    RetrievalPlan,
    RetrievalPlanMode,
)
from .rag_core import (
    fallback_retrieval_plan,
    generation_messages,
    retrieval_plan_adds_signal,
    sanitize_bm25_expansion_terms,
)


_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CONTEXT_LIMIT_RE = re.compile(
    r"^This model's maximum context length is (?P<maximum>\d{1,7}) tokens\. "
    r"However, you requested (?P<output>\d{1,7}) output tokens and your prompt "
    r"contains at least (?P<input>\d{1,7}) input tokens, for a total of at least "
    r"(?P<total>\d{1,7}) tokens\."
)
_MIN_USEFUL_GENERATION_TOKENS = 64
_PLANNER_SYSTEM_PROMPT = (
    "Expand a user's Novo notebook question into retrieval inputs. This is query "
    "expansion, not question restatement. Return exactly one JSON object with keys "
    "semantic_query and bm25_terms. semantic_query must preserve every proper name, "
    "acronym, number, and exact technical term from the current question while adding "
    "two to six close paraphrases of the requested state, action, or relationship in "
    "likely answer-bearing notebook language. bm25_terms must contain four to twelve "
    "short lexical alternatives that add useful search words absent from the current "
    "question; do not repeat question words merely to preserve them. A capitalization-"
    "only rewrite or a list copied from the question is invalid. For example, expand "
    "'What is Mina confused about?' toward 'Mina confusion uncertainty lack of "
    "understanding unsure unclear not sure no clue stuck trouble difficulty' with BM25 "
    "terms such as ['confusion','uncertainty','unclear','unsure','not sure','no clue',"
    "'stuck','trouble','difficulty']. Use recent questions only to resolve references "
    "such as 'that' or 'what about Kevin'. Do not answer the question, explain your "
    "work, cite sources, hypothesize answers, or invent people, projects, experiments, "
    "dates, facts, or conclusions. Do not add any keys."
)
_PLANNER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "novo_retrieval_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "semantic_query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4_000,
                },
                "bm25_terms": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "minItems": 4,
                    "maxItems": 12,
                },
            },
            "required": ["semantic_query", "bm25_terms"],
            "additionalProperties": False,
        },
    },
}


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


def _is_valid_context_overflow(
    response: requests.Response,
    *,
    requested_output_tokens: int,
    configured_context_tokens: int | None,
) -> bool:
    """Return whether a response is a validated vLLM context overflow."""

    if response.status_code != 400 or configured_context_tokens is None:
        return False
    try:
        document = response.json()
    except (requests.RequestException, TypeError, ValueError):
        return False
    if not isinstance(document, Mapping):
        return False
    error = document.get("error", document)
    if not isinstance(error, Mapping):
        return False
    message = error.get("message")
    if not isinstance(message, str) or len(message) > 2_000:
        return False
    match = _CONTEXT_LIMIT_RE.match(message)
    if match is None:
        return False
    maximum = int(match.group("maximum"))
    output_tokens = int(match.group("output"))
    input_tokens = int(match.group("input"))
    total_tokens = int(match.group("total"))
    if (
        maximum != configured_context_tokens
        or output_tokens != requested_output_tokens
        or total_tokens != input_tokens + output_tokens
        or total_tokens <= maximum
    ):
        return False
    return True


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _VllmUsage(BaseModel):
    """The small validated subset of the OpenAI-compatible usage object we consume."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(ge=1, le=2_000_000, strict=True)


class _PlannerResponse(_StrictModel):
    semantic_query: str = Field(min_length=1, max_length=4_000)
    bm25_terms: tuple[str, ...] = Field(min_length=4, max_length=12)


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
    max_tokens: int = Field(default=2048, alias="maxTokens", ge=1, le=65_536)
    temperature: float = Field(default=0.2, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(
        default=None,
        alias="topP",
        gt=0,
        le=1,
        allow_inf_nan=False,
    )
    top_k: int | None = Field(default=None, alias="topK", ge=-1, le=1000, strict=True)
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

    def plan_query(
        self,
        question: str,
        retrieval_question: str,
        *,
        model: str,
    ) -> RetrievalPlan:
        """Create displayable search inputs; any planner failure is non-fatal."""

        fallback = fallback_retrieval_plan(question, retrieval_question)
        backend = self.config.models.get(model)
        if backend is None:
            return fallback
        contextual_question = str(retrieval_question or question).strip()
        if len(contextual_question) > 12_000:
            contextual_question = contextual_question[-12_000:].lstrip()
        planner_document: dict[str, Any] = {
            "current_question": question,
            "recent_questions_and_current_question": contextual_question,
        }

        def request_plan(document: Mapping[str, Any]) -> _PlannerResponse:
            payload: dict[str, Any] = {
                "model": backend.served_model,
                "messages": [
                    {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            document,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "max_tokens": 256,
                "temperature": 0,
                "stream": False,
                "response_format": _PLANNER_RESPONSE_FORMAT,
            }
            if backend.chat_template_kwargs:
                payload["chat_template_kwargs"] = backend.chat_template_kwargs
            response = self._session.post(
                f"{backend.base_url}/v1/chat/completions",
                json=payload,
                timeout=min(30.0, backend.timeout_seconds),
            )
            response.raise_for_status()
            response_document = response.json()
            if not isinstance(response_document, Mapping):
                raise ValueError("planner response must be an object")
            choice = (response_document.get("choices") or [{}])[0]
            content = str((choice.get("message") or {}).get("content") or "").strip()
            if content.startswith("```") and content.endswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
            return _PlannerResponse.model_validate(json.loads(content))

        try:
            planned = request_plan(planner_document)
            bm25_terms = sanitize_bm25_expansion_terms(question, planned.bm25_terms)
            useful = len(bm25_terms) >= 4 and retrieval_plan_adds_signal(
                question, planned.semantic_query, bm25_terms
            )
            if not useful:
                planned = request_plan(
                    {
                        **planner_document,
                        "rejected_plan": planned.model_dump(mode="json"),
                        "correction": (
                            "The previous plan merely copied or restated the current question. "
                            "Replace it with answer-neutral notebook-language expansion that adds "
                            "at least two meaningful semantic tokens and four distinct BM25 "
                            "expansion terms absent from the question."
                        ),
                    }
                )
                bm25_terms = sanitize_bm25_expansion_terms(question, planned.bm25_terms)
                useful = len(bm25_terms) >= 4 and retrieval_plan_adds_signal(
                    question, planned.semantic_query, bm25_terms
                )
            if not useful:
                return fallback
            return RetrievalPlan(
                original_question=question,
                semantic_query=planned.semantic_query,
                bm25_terms=bm25_terms,
                mode=RetrievalPlanMode.PLANNED,
            )
        except (requests.RequestException, AttributeError, IndexError, TypeError, ValueError):
            return fallback

    def generate(
        self,
        question: str,
        hits: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_sources: int,
    ) -> GenerationResult:
        if not hits:
            return GenerationResult(
                answer="The provided documents do not contain this information."
            )
        backend = self.config.models.get(model)
        if backend is None:
            raise ComputeError("MODEL_NOT_APPROVED", "Requested model is not available.")
        payload: dict[str, Any] = {
            "model": backend.served_model,
            "messages": generation_messages(question, hits, max_sources=max_sources),
            "max_tokens": backend.max_tokens,
            "temperature": backend.temperature,
            "stream": False,
        }
        if backend.top_p is not None:
            payload["top_p"] = backend.top_p
        if backend.top_k is not None:
            payload["top_k"] = backend.top_k
        if backend.chat_template_kwargs:
            payload["chat_template_kwargs"] = backend.chat_template_kwargs
        try:
            response = None
            for attempt in range(4):
                response = self._session.post(
                    f"{backend.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=backend.timeout_seconds,
                )
                if response.status_code < 400:
                    break
                current_max_tokens = int(payload["max_tokens"])
                if not _is_valid_context_overflow(
                    response,
                    requested_output_tokens=current_max_tokens,
                    configured_context_tokens=backend.max_model_len,
                ):
                    response.raise_for_status()
                reduced_max_tokens = max(
                    _MIN_USEFUL_GENERATION_TOKENS,
                    current_max_tokens // 2,
                )
                if attempt == 3 or reduced_max_tokens >= current_max_tokens:
                    raise ComputeError(
                        "CONTEXT_LENGTH_EXCEEDED",
                        "The selected notebook context is too large for this model. "
                        "Reduce the number of sources and try again.",
                        retryable=False,
                    )
                payload = {**payload, "max_tokens": reduced_max_tokens}
            assert response is not None
            response.raise_for_status()
            document = response.json()
            if not isinstance(document, Mapping):
                raise ValueError("generation response must be an object")
            choice = (document.get("choices") or [{}])[0]
            answer = str((choice.get("message") or {}).get("content") or "").strip()
            timings = None
            if "usage" in document:
                usage = _VllmUsage.model_validate(document["usage"])
                if backend.max_model_len is not None:
                    timings = QueryTimings(
                        prompt_eval_count=usage.prompt_tokens,
                        num_ctx=backend.max_model_len,
                    )
        except ComputeError:
            raise
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
        return GenerationResult(answer=answer, timings=timings)

    def is_ready(self, model: str) -> bool:
        backend = self.config.models.get(model)
        if backend is None:
            return False
        try:
            response = self._session.get(f"{backend.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(
        self,
        model: str,
        *,
        timeout_seconds: float,
        progress_callback: Callable[[], None] | None = None,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while time.monotonic() < deadline:
            if self.is_ready(model):
                return True
            if progress_callback is not None:
                progress_callback()
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

    def plan_query(
        self,
        question: str,
        retrieval_question: str,
        *,
        model: str,
    ) -> RetrievalPlan:
        del model
        return fallback_retrieval_plan(question, retrieval_question)

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

    def wait_until_ready(
        self,
        model: str,
        *,
        timeout_seconds: float,
        progress_callback: Callable[[], None] | None = None,
    ) -> bool:
        del model, timeout_seconds, progress_callback
        return False


__all__ = [
    "HttpModelBackend",
    "ModelBackendDocument",
    "UnavailableModelBackend",
    "load_model_backend_document",
]
