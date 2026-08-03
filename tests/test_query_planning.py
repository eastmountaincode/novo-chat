from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from novo_chat.compute import IndexRepository, LoadedIndexArtifact
from novo_chat.protocol import (
    JobProgressDetail,
    NotebookScope,
    QueryProgressStage,
    QueryStrategy,
    RetrievalPlan,
    RetrievalPlanMode,
)


class PlannerBackend:
    def __init__(self, plan: RetrievalPlan | None = None, *, fail_planner: bool = False):
        self.plan = plan
        self.fail_planner = fail_planner
        self.generated_question = ""
        self.generated_hits: list[dict[str, Any]] = []
        self.embedded_queries: list[str] = []

    def embed_documents(self, texts: Sequence[str], *, scope: NotebookScope) -> np.ndarray:
        del texts, scope
        raise AssertionError("documents are not embedded in these tests")

    def embed_query(self, text: str) -> np.ndarray:
        self.embedded_queries.append(text)
        lowered = text.lower()
        if "her2" in lowered:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if "binding affinity" in lowered:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([1.0, 1.0], dtype=np.float32)

    def plan_query(
        self,
        question: str,
        retrieval_question: str,
        *,
        model: str,
    ) -> RetrievalPlan:
        del question, retrieval_question, model
        if self.fail_planner:
            raise RuntimeError("planner unavailable")
        assert self.plan is not None
        return self.plan

    def generate(
        self,
        question: str,
        hits: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_sources: int,
    ) -> str:
        del model, max_sources
        self.generated_question = question
        self.generated_hits = [dict(hit) for hit in hits]
        return "Grounded answer [1]"


def scope() -> NotebookScope:
    return NotebookScope(
        notebook_id="notebook-a",
        content_revision="revision-a",
        index_schema_version="index-v1",
    )


def chunk(page_id: str, indexed_text: str, *, chunk_idx: int = 0) -> dict[str, Any]:
    return {
        "file": f"{page_id}.md",
        "title": page_id,
        "chunk_idx": chunk_idx,
        "text": indexed_text,
        "indexed_text": indexed_text,
        "metadata": {},
        "page_id": page_id,
        "notebook_id": "notebook-a",
        "notebook": "notebook-a",
        "source_url": f"/?page={page_id}",
    }


def artifact(chunks: Sequence[dict[str, Any]], vectors: Sequence[Sequence[float]]) -> LoadedIndexArtifact:
    return LoadedIndexArtifact(
        scope=scope(),
        artifact_id="a" * 64,
        chunks=tuple(chunks),
        vectors=np.asarray(vectors, dtype=np.float32),
    )


def test_original_and_semantic_queries_are_both_real_dense_signals(tmp_path) -> None:
    plan = RetrievalPlan(
        original_question="What is the HER2 status?",
        semantic_query="binding affinity experiments",
        bm25_terms=(),
    )
    backend = PlannerBackend(plan)
    repository = IndexRepository(tmp_path, backend)
    chunks = [
        chunk("page-original", "HER2 receptor status"),
        chunk("page-semantic", "binding affinity experiments"),
        chunk("page-distractor-a", "unrelated alpha"),
        chunk("page-distractor-b", "unrelated beta"),
    ]

    hits = repository._retrieve(
        chunks,
        np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32),
        "What is the HER2 status?",
        plan,
        strategy=QueryStrategy.SEMANTIC,
        count=4,
    )

    assert {hit["page_id"] for hit in hits[:2]} == {"page-original", "page-semantic"}
    assert backend.embedded_queries == [
        "What is the HER2 status?",
        "binding affinity experiments",
    ]


def test_equivalent_dense_queries_are_embedded_only_once(tmp_path) -> None:
    plan = RetrievalPlan(
        original_question="HER2 status",
        semantic_query="  her2 STATUS  ",
        bm25_terms=("HER2",),
    )
    backend = PlannerBackend(plan)
    repository = IndexRepository(tmp_path, backend)

    repository._retrieve(
        [
            chunk("page-original", "HER2 receptor status"),
            chunk("page-other", "other"),
        ],
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        "HER2 status",
        plan,
        strategy=QueryStrategy.SEMANTIC,
        count=2,
    )

    assert backend.embedded_queries == ["HER2 status"]


def test_planned_bm25_terms_add_exact_keyword_retrieval(tmp_path) -> None:
    plan = RetrievalPlan(
        original_question="Which assay is blocked?",
        semantic_query="blocked assay troubleshooting",
        bm25_terms=("Q5",),
    )
    backend = PlannerBackend(plan)
    repository = IndexRepository(tmp_path, backend)
    chunks = [
        chunk("page-alpha", "alpha assay"),
        chunk("page-beta", "beta assay"),
        chunk("page-q5", "Q5 primer optimization failure"),
        chunk("page-gamma", "gamma assay"),
    ]

    hits = repository._retrieve(
        chunks,
        np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32),
        "Which assay is blocked?",
        plan,
        strategy=QueryStrategy.LEXICAL,
        count=4,
    )

    assert hits[0]["page_id"] == "page-q5"
    assert hits[0]["bm25"] > 0


def test_query_preserves_original_for_generation_and_reports_real_stages(tmp_path) -> None:
    question = "What is the HER2 status?"
    plan = RetrievalPlan(
        original_question=question,
        semantic_query="binding affinity experiments",
        bm25_terms=("HER2", "binding affinity"),
    )
    backend = PlannerBackend(plan)
    repository = IndexRepository(tmp_path, backend)
    active = artifact(
        [
            chunk("page-original", "HER2 receptor status"),
            chunk("page-semantic", "binding affinity experiments"),
            chunk("page-q5", "Q5 primer optimization"),
            chunk("page-other", "unrelated notes"),
        ],
        [[1, 0], [0, 1], [0.7, 0.7], [-1, 0]],
    )
    updates: list[JobProgressDetail] = []

    result = repository.query(
        [active],
        question=question,
        retrieval_question=f"Earlier question\n\n{question}",
        model="model:a",
        strategy=QueryStrategy.HYBRID,
        max_sources=3,
        retrieval_top_k=4,
        progress_callback=updates.append,
    )

    assert backend.generated_question == question
    assert result.retrieval_plan == plan
    assert [update.stage for update in updates] == [
        QueryProgressStage.SEARCHING,
        QueryProgressStage.ANSWERING,
    ]
    assert updates[-1].retrieved_count == 4
    assert len(result.citations) == 4


def test_planner_exception_falls_back_to_contextual_query(tmp_path) -> None:
    backend = PlannerBackend(fail_planner=True)
    repository = IndexRepository(tmp_path, backend)

    plan = repository._query_plan(
        question="What about Kevin?",
        retrieval_question="What projects is Andrew working on?\n\nWhat about Kevin?",
        model="model:a",
        use_model=True,
    )

    assert plan.mode is RetrievalPlanMode.FALLBACK
    assert plan.original_question == "What about Kevin?"
    assert plan.semantic_query.endswith("What about Kevin?")
    assert "kevin" in plan.bm25_terms
