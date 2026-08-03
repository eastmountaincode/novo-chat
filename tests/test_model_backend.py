from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import requests
from pydantic import ValidationError

from novo_chat.compute import ComputeError
from novo_chat.model_backend import (
    HttpModelBackend,
    ModelBackendDocument,
    UnavailableModelBackend,
    load_model_backend_document,
)
from novo_chat.protocol import NotebookScope, RetrievalPlanMode


NO_INFORMATION = "The provided documents do not contain this information."


def valid_config() -> dict[str, Any]:
    return {
        "version": 1,
        "embedding": {
            "baseUrl": "http://127.0.0.1:11434",
            "model": "nomic-embed-text",
            "maxCharacters": 128,
            "documentKeepAlive": "10m",
            "queryKeepAlive": "0",
            "timeoutSeconds": 12,
        },
        "models": {
            "model:a": {
                "baseUrl": "http://localhost:8000",
                "servedModel": "served-model",
                "maxTokens": 512,
                "timeoutSeconds": 30,
                "chatTemplateKwargs": {"enable_thinking": False},
            }
        },
    }


class FakeResponse:
    def __init__(self, payload: Any = None, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(
        self,
        *,
        posts: list[FakeResponse | Exception] | None = None,
        gets: list[FakeResponse | Exception] | None = None,
    ) -> None:
        self.trust_env = True
        self.posts = deque(posts or [])
        self.gets = deque(gets or [])
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def post(self, url: str, *, json: Any, timeout: float) -> FakeResponse:
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        result = self.posts.popleft()
        if isinstance(result, Exception):
            raise result
        return result

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.get_calls.append({"url": url, "timeout": timeout})
        result = self.gets.popleft()
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://10.0.0.8:11434",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?target=elsewhere",
        "http://127.0.0.1:11434/#fragment",
        "http://127.0.0.1:not-a-port",
        "http://127.0.0.1",
    ],
)
def test_configuration_rejects_non_loopback_or_unsafe_backend_urls(unsafe_url: str) -> None:
    document = valid_config()
    document["embedding"]["baseUrl"] = unsafe_url
    with pytest.raises(ValidationError):
        ModelBackendDocument.model_validate(document)

    document = valid_config()
    document["models"]["model:a"]["baseUrl"] = unsafe_url
    with pytest.raises(ValidationError):
        ModelBackendDocument.model_validate(document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update({"unknown": "docker run --privileged"}),
        lambda document: document["embedding"].update({"headers": {"Authorization": "secret"}}),
        lambda document: document["models"]["model:a"].update({"command": ["docker", "run"]}),
        lambda document: document.update({"version": 2}),
        lambda document: document.update({"models": {}}),
        lambda document: document["models"]["model:a"].update(
            {"chatTemplateKwargs": {"nonfinite": float("nan")}}
        ),
        lambda document: document["models"]["model:a"].update(
            {"chatTemplateKwargs": {"oversized": "x" * 8192}}
        ),
        lambda document: document.update(
            {
                "models": {
                    "unsafe/model": document["models"]["model:a"],
                }
            }
        ),
        lambda document: document.update(
            {
                "models": {
                    "m" * 129: document["models"]["model:a"],
                }
            }
        ),
    ],
)
def test_configuration_rejects_unknown_or_unsupported_values(mutation) -> None:
    document = valid_config()
    mutation(document)

    with pytest.raises(ValidationError):
        ModelBackendDocument.model_validate(document)


def test_configuration_accepts_explicit_ipv4_ipv6_and_localhost_loopback() -> None:
    document = valid_config()
    document["embedding"]["baseUrl"] = "http://[::1]:11434"
    document["models"]["model:a"]["baseUrl"] = "http://localhost:8000/"

    parsed = ModelBackendDocument.model_validate(document)

    assert parsed.embedding.base_url == "http://[::1]:11434"
    assert parsed.models["model:a"].base_url == "http://localhost:8000"


def test_model_display_metadata_is_optional_typed_and_exported() -> None:
    legacy_backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()))
    legacy_details = legacy_backend.model_details["model:a"]
    assert legacy_details.model_dump(mode="json", by_alias=True, exclude_none=True) == {
        "maxTokens": 512,
    }

    document = valid_config()
    document["models"]["model:a"].update(
        {
            "modelSize": " 122B ",
            "maxModelLen": 262_144,
            "thinking": "enabled",
            "totalVramGb": 192,
        }
    )
    backend = HttpModelBackend(ModelBackendDocument.model_validate(document))

    assert backend.model_details["model:a"].model_dump(mode="json", by_alias=True) == {
        "modelSize": "122B",
        "maxTokens": 512,
        "maxModelLen": 262_144,
        "thinking": "enabled",
        "totalVramGb": 192.0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("modelSize", "   "),
        ("modelSize", "122B\nunsafe"),
        ("modelSize", 122),
        ("maxModelLen", 0),
        ("maxModelLen", True),
        ("maxModelLen", "262144"),
        ("thinking", ""),
        ("thinking", "enabled\tunsafe"),
        ("totalVramGb", 0),
        ("totalVramGb", True),
        ("totalVramGb", "192"),
        ("totalVramGb", float("nan")),
        ("totalVramGb", float("inf")),
    ],
)
def test_model_display_metadata_rejects_ambiguous_or_unsafe_values(field: str, value: Any) -> None:
    document = valid_config()
    document["models"]["model:a"][field] = value

    with pytest.raises(ValidationError):
        ModelBackendDocument.model_validate(document)


def test_load_configuration_requires_an_absolute_valid_json_file(tmp_path: Path) -> None:
    relative = Path("model-backends.json")
    with pytest.raises(ValueError, match="must be absolute"):
        load_model_backend_document(relative)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable or invalid"):
        load_model_backend_document(malformed)

    configured = tmp_path / "model-backends.json"
    configured.write_text(json.dumps(valid_config()), encoding="utf-8")
    loaded = load_model_backend_document(configured)
    assert tuple(loaded.models) == ("model:a",)


def test_embedding_generation_and_readiness_use_only_the_fake_session() -> None:
    session = FakeSession(
        posts=[
            FakeResponse({"embedding": [3.0, 4.0]}),
            FakeResponse({"embedding": [0.0, 5.0]}),
            FakeResponse({"embedding": [8.0, 6.0]}),
            FakeResponse(
                {
                    "choices": [{"message": {"content": "Grounded answer [1]"}}],
                    "usage": {
                        "prompt_tokens": 12_345,
                        "completion_tokens": 20,
                        "total_tokens": 12_365,
                    },
                }
            ),
        ],
        gets=[FakeResponse(status_code=503), FakeResponse(status_code=200)],
    )
    configuration = valid_config()
    configuration["models"]["model:a"]["maxModelLen"] = 262_144
    backend = HttpModelBackend(ModelBackendDocument.model_validate(configuration), session=session)
    scope = NotebookScope(
        notebook_id="notebook-a",
        content_revision="sha256:" + "a" * 64,
        index_schema_version="novo-chat-index-v1",
    )

    vectors = backend.embed_documents(["a" * 140, "second"], scope=scope)
    query = backend.embed_query("find this")
    hits = [
        {
            "source_idx": 1,
            "file": "page-a.md",
            "page_id": "page-a",
            "notebook": "notebook-a",
            "title": "Experiment",
            "metadata": {"created": "2026-07-01T00:00:00Z"},
            "text": "Grounding text",
        }
    ]
    result = backend.generate("What happened?", hits, model="model:a", max_sources=1)
    with patch("novo_chat.model_backend.time.sleep", return_value=None):
        ready = backend.wait_until_ready("model:a", timeout_seconds=1)

    assert session.trust_env is False
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32))
    np.testing.assert_array_equal(query, np.asarray([8.0, 6.0], dtype=np.float32))
    assert result.answer == "Grounded answer [1]"
    assert result.model_dump(mode="json", by_alias=True)["timings"] == {
        "prompt_eval_count": 12_345,
        "num_ctx": 262_144,
    }
    assert ready is True
    assert backend.is_ready("unknown-model") is False

    first_embedding = session.post_calls[0]
    assert first_embedding["url"] == "http://127.0.0.1:11434/api/embeddings"
    assert first_embedding["json"] == {
        "model": "nomic-embed-text",
        "prompt": "search_document: " + "a" * 128,
        "keep_alive": "10m",
    }
    assert first_embedding["timeout"] == 12
    assert session.post_calls[2]["json"]["prompt"] == "search_query: find this"
    assert session.post_calls[2]["json"]["keep_alive"] == "0"

    generation = session.post_calls[3]
    assert generation["url"] == "http://localhost:8000/v1/chat/completions"
    assert generation["timeout"] == 30
    assert generation["json"]["model"] == "served-model"
    assert generation["json"]["max_tokens"] == 512
    assert generation["json"]["temperature"] == 0.2
    assert generation["json"]["stream"] is False
    assert generation["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert generation["json"]["messages"][1] == {"role": "user", "content": "What happened?"}
    assert "SOURCE [1]" in generation["json"]["messages"][0]["content"]
    assert session.get_calls == [
        {"url": "http://localhost:8000/health", "timeout": 2.0},
        {"url": "http://localhost:8000/health", "timeout": 2.0},
    ]


def test_query_planner_uses_a_small_structured_call_and_preserves_original_question() -> None:
    session = FakeSession(
        posts=[
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "semantic_query": "Defne projects blockers troubleshooting",
                                        "bm25_terms": [
                                            "Defne",
                                            "blocked",
                                            "blockers",
                                            "difficulty",
                                            "stuck",
                                            "trouble",
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        ]
    )
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)

    plan = backend.plan_query(
        "What is Defne having trouble with?",
        "What projects is Andrew working on?\n\nWhat is Defne having trouble with?",
        model="model:a",
    )

    assert plan.original_question == "What is Defne having trouble with?"
    assert plan.semantic_query == "Defne projects blockers troubleshooting"
    assert plan.bm25_terms == ("blocked", "blockers", "difficulty", "stuck")
    assert plan.mode is RetrievalPlanMode.PLANNED
    request = session.post_calls[0]
    assert request["url"] == "http://localhost:8000/v1/chat/completions"
    assert request["timeout"] == 30
    assert request["json"]["max_tokens"] == 256
    assert request["json"]["temperature"] == 0
    assert request["json"]["stream"] is False
    assert request["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["json"]["response_format"]["type"] == "json_schema"
    assert request["json"]["response_format"]["json_schema"]["strict"] is True
    assert (
        request["json"]["response_format"]["json_schema"]["schema"]["properties"]
        ["bm25_terms"]["maxItems"]
        == 12
    )
    assert (
        request["json"]["response_format"]["json_schema"]["schema"]["properties"]
        ["bm25_terms"]["minItems"]
        == 4
    )
    assert set(
        request["json"]["response_format"]["json_schema"]["schema"]["required"]
    ) == {"semantic_query", "bm25_terms"}
    assert (
        request["json"]["response_format"]["json_schema"]["schema"]
        ["additionalProperties"]
        is False
    )
    assert "query expansion, not question restatement" in request["json"]["messages"][0]["content"]
    assert "capitalization-only rewrite" in request["json"]["messages"][0]["content"]
    assert "Do not answer the question" in request["json"]["messages"][0]["content"]
    planner_input = json.loads(request["json"]["messages"][1]["content"])
    assert planner_input["current_question"] == "What is Defne having trouble with?"
    assert planner_input["recent_questions_and_current_question"].endswith(
        "What is Defne having trouble with?"
    )


def test_query_planner_accepts_json_fences_and_truncates_history_from_the_front() -> None:
    content = """```json
{"semantic_query":"Kevin projects status","bm25_terms":["Kevin","projects","status","work","tasks"]}
```"""
    session = FakeSession(
        posts=[FakeResponse({"choices": [{"message": {"content": content}}]})]
    )
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)
    current_question = "What about Kevin?"
    long_history = "old context " * 2_000 + current_question

    plan = backend.plan_query(
        current_question,
        long_history,
        model="model:a",
    )

    assert plan.mode is RetrievalPlanMode.PLANNED
    assert plan.semantic_query == "Kevin projects status"
    assert plan.bm25_terms == ("projects", "status", "work", "tasks")
    planner_input = json.loads(session.post_calls[0]["json"]["messages"][1]["content"])
    contextual = planner_input["recent_questions_and_current_question"]
    assert len(contextual) <= 12_000
    assert contextual.endswith(current_question)


def test_query_planner_repairs_a_valid_but_retrieval_noop_response() -> None:
    session = FakeSession(
        posts=[
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "semantic_query": "What is Arya confused about?",
                                        "bm25_terms": ["What", "Arya", "confused", "about"],
                                    }
                                )
                            }
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "semantic_query": (
                                            "Arya confusion uncertainty lack of understanding "
                                            "unsure unclear not sure no clue stuck trouble difficulty"
                                        ),
                                        "bm25_terms": [
                                            "Arya",
                                            "confused",
                                            "confusion",
                                            "uncertainty",
                                            "unclear",
                                            "not sure",
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                }
            ),
        ]
    )
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)

    plan = backend.plan_query(
        "What is Arya confused about?",
        "What is Arya confused about?",
        model="model:a",
    )

    assert plan.mode is RetrievalPlanMode.PLANNED
    assert plan.semantic_query.startswith("Arya confusion uncertainty")
    assert plan.bm25_terms == ("confusion", "uncertainty", "unclear", "not sure")
    assert len(session.post_calls) == 2
    assert (
        session.post_calls[1]["json"]["response_format"]
        == session.post_calls[0]["json"]["response_format"]
    )
    repair_input = json.loads(session.post_calls[1]["json"]["messages"][1]["content"])
    assert repair_input["rejected_plan"] == {
        "semantic_query": "What is Arya confused about?",
        "bm25_terms": ["What", "Arya", "confused", "about"],
    }
    assert "merely copied or restated" in repair_input["correction"]


def test_query_planner_falls_back_after_two_retrieval_noop_responses() -> None:
    first_noop = FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "semantic_query": "What is Arya confused about?",
                                "bm25_terms": ["What", "Arya", "confused", "about"],
                            }
                        )
                    }
                }
            ]
        }
    )
    second_noop = FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "semantic_query": "what is ARYA confused about",
                                "bm25_terms": ["WHAT", "ARYA", "CONFUSED", "ABOUT"],
                            }
                        )
                    }
                }
            ]
        }
    )
    session = FakeSession(posts=[first_noop, second_noop])
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)

    plan = backend.plan_query(
        "What is Arya confused about?",
        "What is Arya confused about?",
        model="model:a",
    )

    assert plan.mode is RetrievalPlanMode.FALLBACK
    assert plan.semantic_query == "What is Arya confused about?"
    assert len(session.post_calls) == 2


def test_query_planner_rejects_punctuation_padded_bm25_terms() -> None:
    first_noop = FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "semantic_query": "What is Arya confused about?",
                                "bm25_terms": ["What", "Arya", "confused", "about"],
                            }
                        )
                    }
                }
            ]
        }
    )
    padded_response = FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "semantic_query": "Arya uncertainty and lack of understanding",
                                "bm25_terms": [
                                    "uncertainty",
                                    "un/certainty",
                                    "uncer_tainty",
                                    "uncerta.inty",
                                ],
                            }
                        )
                    }
                }
            ]
        }
    )
    session = FakeSession(posts=[first_noop, padded_response])
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)

    plan = backend.plan_query(
        "What is Arya confused about?",
        "What is Arya confused about?",
        model="model:a",
    )

    assert plan.mode is RetrievalPlanMode.FALLBACK
    assert plan.semantic_query == "What is Arya confused about?"
    assert len(session.post_calls) == 2
    repair_input = json.loads(session.post_calls[1]["json"]["messages"][1]["content"])
    assert "four distinct BM25 expansion terms" in repair_input["correction"]


@pytest.mark.parametrize(
    "response",
    [
        requests.Timeout("planner timeout"),
        FakeResponse({"choices": [{"message": {"content": "{not-json"}}]}),
        FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "semantic_query": "",
                                    "bm25_terms": [],
                                }
                            )
                        }
                    }
                ]
            }
        ),
        FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "semantic_query": "query",
                                    "bm25_terms": [],
                                    "rationale": "must never be exposed",
                                }
                            )
                        }
                    }
                ]
            }
        ),
    ],
)
def test_query_planner_failures_fall_back_without_failing_the_query(response: Any) -> None:
    session = FakeSession(posts=[response])
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)

    plan = backend.plan_query(
        "What is definitely having trouble with?",
        "Previous question\n\nWhat is definitely having trouble with?",
        model="model:a",
    )

    assert plan.original_question == "What is definitely having trouble with?"
    assert plan.semantic_query.endswith("What is definitely having trouble with?")
    assert plan.mode is RetrievalPlanMode.FALLBACK
    assert "definitely" in plan.bm25_terms
    assert "trouble" in plan.bm25_terms


def test_no_hit_generation_returns_exact_answer_without_network() -> None:
    session = FakeSession()
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)

    result = backend.generate("Unknown?", [], model="model:a", max_sources=4)

    assert result.answer == NO_INFORMATION
    assert result.timings is None
    assert session.post_calls == []


def test_generation_without_usage_remains_backward_compatible() -> None:
    session = FakeSession(
        posts=[FakeResponse({"choices": [{"message": {"content": "Grounded answer [1]"}}]})]
    )
    configuration = valid_config()
    configuration["models"]["model:a"]["maxModelLen"] = 262_144
    backend = HttpModelBackend(ModelBackendDocument.model_validate(configuration), session=session)

    result = backend.generate(
        "Question",
        [{"source_idx": 1, "text": "source"}],
        model="model:a",
        max_sources=1,
    )

    assert result.answer == "Grounded answer [1]"
    assert result.timings is None


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 0},
        {"prompt_tokens": -1},
        {"prompt_tokens": True},
        {"prompt_tokens": 12.5},
        {"prompt_tokens": "123"},
        {"prompt_tokens": 2_000_001},
    ],
)
def test_generation_rejects_malformed_vllm_usage(usage: Any) -> None:
    session = FakeSession(
        posts=[
            FakeResponse(
                {
                    "choices": [{"message": {"content": "Grounded answer [1]"}}],
                    "usage": usage,
                }
            )
        ]
    )
    configuration = valid_config()
    configuration["models"]["model:a"]["maxModelLen"] = 262_144
    backend = HttpModelBackend(ModelBackendDocument.model_validate(configuration), session=session)

    with pytest.raises(ComputeError) as invalid:
        backend.generate(
            "Question",
            [{"source_idx": 1, "text": "source"}],
            model="model:a",
            max_sources=1,
        )

    assert invalid.value.code == "GENERATION_UNAVAILABLE"
    assert invalid.value.retryable is True


def test_unknown_model_and_unavailable_backend_fail_closed() -> None:
    session = FakeSession()
    backend = HttpModelBackend(ModelBackendDocument.model_validate(valid_config()), session=session)
    hits = [{"source_idx": 1, "text": "source"}]

    with pytest.raises(ComputeError) as unknown:
        backend.generate("Question", hits, model="unknown-model", max_sources=1)
    assert unknown.value.code == "MODEL_NOT_APPROVED"
    assert session.post_calls == []

    unavailable = UnavailableModelBackend()
    with pytest.raises(ComputeError) as embedding:
        unavailable.embed_query("question")
    assert embedding.value.code == "MODEL_BACKEND_UNAVAILABLE"
    assert embedding.value.retryable is True
    assert unavailable.is_ready("model:a") is False
    assert unavailable.wait_until_ready("model:a", timeout_seconds=0) is False
