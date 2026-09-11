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
from novo_chat.rag_core import retrieval_plan_adds_signal, sanitize_bm25_expansion_terms


class PlannerBackend:
    def __init__(self, plan: RetrievalPlan | None = None, *, fail_planner: bool = False):
        self.plan = plan
        self.fail_planner = fail_planner
        self.generated_question = ""
        self.generated_hits: list[dict[str, Any]] = []
        self.embedded_queries: list[str] = []

    def embed_documents(self, texts: Sequence[str], *, scope: NotebookScope, progress_callback=None) -> np.ndarray:
        del texts, scope, progress_callback
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


def test_copied_plan_is_not_a_retrieval_signal_and_terms_are_removed() -> None:
    question = "What is Arya confused about?"

    assert retrieval_plan_adds_signal(
        question,
        "What is Arya confused about?",
        ("Arya", "confused"),
    ) is False
    assert sanitize_bm25_expansion_terms(
        question,
        (
            "Arya",
            "confused",
            "uncertainty",
            "uncertainty!",
            "uncertainty?",
            "un-certainty",
            "uncer-tainty",
            "uncer‑tainty",
            "uncerta-inty",
            "un/certainty",
            "uncer_tainty",
            "uncerta.inty",
            "No clue",
            "UNCERTAINTY",
        ),
    ) == ("uncertainty", "No clue")

    assert sanitize_bm25_expansion_terms(
        "What is the HER2 status?",
        ("HER-2", "binding affinity", "binding-affinity", "Q5"),
    ) == ("HER-2", "binding affinity", "Q5")

    assert sanitize_bm25_expansion_terms(
        "Which histone mark changed?",
        ("H3 K27", "H3K27", "HER-2", "HER2"),
    ) == ("H3 K27", "H3K27", "HER-2", "HER2")


def test_punctuation_fragments_do_not_manufacture_semantic_signal() -> None:
    question = "What is Arya confused about?"

    assert retrieval_plan_adds_signal(
        question,
        "un-certainty uncer-tainty",
        (),
    ) is False
    assert retrieval_plan_adds_signal(question, "binding affinity", ()) is True


def test_semantic_and_bm25_expansion_add_meaningful_retrieval_tokens() -> None:
    question = "What is Arya confused about?"

    assert retrieval_plan_adds_signal(
        question,
        "Arya confusion uncertainty lack of understanding unsure unclear",
        ("confusion", "uncertainty", "not sure", "no clue"),
    ) is True


def test_original_and_semantic_queries_are_both_real_dense_signals(tmp_path) -> None:
    question = "What is the HER2 status?"
    plan = RetrievalPlan(
        original_question=question,
        semantic_query="binding affinity experiments",
        bm25_terms=(),
    )
    backend = PlannerBackend(plan)
    repository = IndexRepository(tmp_path, backend)
    chunks = [
        chunk("page-original", "HER2 receptor status"),
        *[
            chunk(f"page-distractor-{index}", f"unrelated distractor {index}")
            for index in range(21)
        ],
        chunk("page-semantic", "binding affinity experiments"),
    ]

    vectors = np.asarray(
        [[1, 0], *([[0.5, -0.5]] * 21), [0, 1]],
        dtype=np.float32,
    )
    baseline_backend = PlannerBackend(
        RetrievalPlan(
            original_question=question,
            semantic_query=question,
            bm25_terms=(),
        )
    )
    baseline_hits = IndexRepository(tmp_path / "baseline", baseline_backend)._retrieve(
        chunks,
        vectors,
        question,
        baseline_backend.plan,
        strategy=QueryStrategy.SEMANTIC,
        count=len(chunks),
    )
    hits = repository._retrieve(
        chunks,
        vectors,
        question,
        plan,
        strategy=QueryStrategy.SEMANTIC,
        count=16,
    )

    original_hit = next(hit for hit in hits if hit["page_id"] == "page-original")
    semantic_hit = next(hit for hit in hits if hit["page_id"] == "page-semantic")
    baseline_original_hit = next(
        hit for hit in baseline_hits if hit["page_id"] == "page-original"
    )
    baseline_semantic_hit = next(
        hit for hit in baseline_hits if hit["page_id"] == "page-semantic"
    )
    assert original_hit["score"] >= baseline_original_hit["score"]
    assert semantic_hit["score"] > baseline_semantic_hit["score"]
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
        *[
            chunk(f"page-generic-{index}", f"generic assay {index}")
            for index in range(10)
        ],
        chunk("page-q5", "Q5 primer optimization failure"),
    ]

    hits = repository._retrieve(
        chunks,
        np.asarray([[1, 0]] * len(chunks), dtype=np.float32),
        "Which assay is blocked?",
        plan,
        strategy=QueryStrategy.LEXICAL,
        count=8,
    )

    q5_hit = next(hit for hit in hits if hit["page_id"] == "page-q5")
    assert q5_hit["bm25"] > 0


def test_exact_lexical_overlap_survives_nonpositive_bm25_scores(tmp_path) -> None:
    question = "HER2"
    plan = RetrievalPlan(
        original_question=question,
        semantic_query=question,
        bm25_terms=(),
    )

    for case_index, chunks in enumerate(
        (
            [chunk("page-exact", "HER2")],
            [chunk("page-exact", "HER2"), chunk("page-other", "unrelated")],
        )
    ):
        hits = IndexRepository(
            tmp_path / f"nonpositive-bm25-{case_index}",
            PlannerBackend(plan),
        )._retrieve(
            chunks,
            np.asarray([[1, 0]] * len(chunks), dtype=np.float32),
            question,
            plan,
            strategy=QueryStrategy.LEXICAL,
            count=len(chunks),
        )

        assert [hit["page_id"] for hit in hits] == ["page-exact"]


def test_scaffolding_overlap_cannot_outrank_small_corpus_exact_match(tmp_path) -> None:
    question = "What is the HER2 status?"
    plan = RetrievalPlan(
        original_question=question,
        semantic_query=question,
        bm25_terms=(),
    )
    chunks = [
        chunk("page-generic", "what is the result"),
        chunk("page-exact", "HER2"),
    ]
    vectors = np.asarray([[1, 0], [-1, 0]], dtype=np.float32)

    for strategy in (QueryStrategy.LEXICAL, QueryStrategy.HYBRID):
        hits = IndexRepository(
            tmp_path / f"scaffolding-{strategy.value}",
            PlannerBackend(plan),
        )._retrieve(
            chunks,
            vectors,
            question,
            plan,
            strategy=strategy,
            count=1,
        )

        assert [hit["page_id"] for hit in hits] == ["page-exact"]


def test_bad_lexical_expansion_cannot_outrank_an_exact_original_hit(tmp_path) -> None:
    expansion_terms = tuple(f"novelterm{index}" for index in range(12))
    question = "HER2 receptor status"
    plan = RetrievalPlan(
        original_question=question,
        semantic_query=" ".join(expansion_terms),
        bm25_terms=expansion_terms,
    )
    backend = PlannerBackend(plan)
    repository = IndexRepository(tmp_path, backend)
    chunks = [
        chunk("page-exact", "HER2 receptor status"),
        chunk("page-bad-expansion", " ".join(expansion_terms)),
        chunk("page-distractor-a", "unrelated alpha notes"),
        chunk("page-distractor-b", "unrelated beta notes"),
    ]

    hits = repository._retrieve(
        chunks,
        np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32),
        question,
        plan,
        strategy=QueryStrategy.LEXICAL,
        count=1,
    )

    assert [hit["page_id"] for hit in hits] == ["page-exact"]


def test_bad_hybrid_expansion_cannot_stack_dense_and_lexical_credit(tmp_path) -> None:
    expansion_terms = tuple(f"novelterm{index}" for index in range(12))
    question = "HER2 receptor status"
    planned = RetrievalPlan(
        original_question=question,
        semantic_query=f"binding affinity {' '.join(expansion_terms)}",
        bm25_terms=expansion_terms,
    )
    baseline = RetrievalPlan(
        original_question=question,
        semantic_query=question,
        bm25_terms=(),
    )
    chunks = [
        chunk("page-exact", question),
        chunk("page-expansion", planned.semantic_query),
        *[
            chunk(f"page-distractor-{index}", f"unrelated distractor {index}")
            for index in range(240)
        ],
    ]
    vectors = np.asarray(
        [[-1, -1], [0, 1], *([[1, 0.1]] * 240)],
        dtype=np.float32,
    )

    baseline_hits = IndexRepository(
        tmp_path / "baseline-hybrid",
        PlannerBackend(baseline),
    )._retrieve(
        chunks,
        vectors,
        question,
        baseline,
        strategy=QueryStrategy.HYBRID,
        count=10,
    )
    planned_hits = IndexRepository(
        tmp_path / "planned-hybrid",
        PlannerBackend(planned),
    )._retrieve(
        chunks,
        vectors,
        question,
        planned,
        strategy=QueryStrategy.HYBRID,
        count=10,
    )

    assert planned_hits[0]["page_id"] == "page-exact"
    assert planned_hits[0]["score"] == baseline_hits[0]["score"]
    assert "page-expansion" in [hit["page_id"] for hit in planned_hits]
    assert "page-expansion" not in [hit["page_id"] for hit in baseline_hits]


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
    assert updates[-1].retrieved_count == 3
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
