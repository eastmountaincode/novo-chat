from __future__ import annotations

from novo_chat.rag_core import (
    check_citations,
    chunk_text,
    diversify_by_page,
    generation_messages,
    normalize_citation_glyphs,
    tokenize,
)


NO_INFORMATION = "The provided documents do not contain this information."


def test_search_normalization_chunk_overlap_and_page_diversification() -> None:
    assert tokenize("5 µM‑EGTA") == ["5", "um", "egta"]

    chunks = chunk_text("one two three four five six seven", words=4, overlap=2)
    assert chunks == ["one two three four", "three four five six", "five six seven"]

    metadata = [
        {"page_id": "page-a"},
        {"page_id": "page-a"},
        {"page_id": "page-b"},
        {"page_id": "page-c"},
    ]
    ranked = [(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)]
    assert diversify_by_page(ranked, metadata, 3, max_per_page=1) == [
        (0, 1.0),
        (2, 0.8),
        (3, 0.7),
    ]


def test_generation_messages_are_bounded_and_keep_source_ids() -> None:
    hits = [
        {
            "source_idx": 1,
            "file": "page-a.md",
            "page_id": "page-a",
            "notebook": "notebook-a",
            "title": "First experiment",
            "metadata": {"created": "2026-07-01T00:00:00Z"},
            "tags": ["assay"],
            "attachments": ["results.csv"],
            "text": "First source text.",
        },
        {
            "source_idx": 2,
            "file": "page-b.md",
            "page_id": "page-b",
            "notebook": "notebook-b",
            "title": "Second experiment",
            "metadata": {},
            "text": "Second source text.",
        },
    ]

    messages = generation_messages("What happened?", hits, max_sources=1)

    assert messages[0]["role"] == "system"
    assert "SOURCE [1]" in messages[0]["content"]
    assert "First source text." in messages[0]["content"]
    assert "results.csv" in messages[0]["content"]
    assert "SOURCE [2]" not in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "What happened?"}


def test_citation_normalization_and_validation() -> None:
    normalized, normalized_sources = normalize_citation_glyphs(
        "The assay worked【1†source】 and reproduced【2, 3】."
    )

    assert normalized == "The assay worked [1] and reproduced [2, 3]."
    assert normalized_sources == [1, 2, 3]
    assert check_citations(normalized, valid_source_ids=[1, 2, 3]) == {
        "ok": True,
        "cited_sources": [1, 2, 3],
        "missing_sources": [],
        "nonstandard_sources": [],
        "valid_sources": [1, 2, 3],
    }

    missing = check_citations("Unsupported claim [4].", valid_source_ids=[1, 2, 3])
    assert missing["ok"] is False
    assert missing["missing_sources"] == [4]

    uncited = check_citations("A factual answer without a source.", valid_source_ids=[1])
    assert uncited["ok"] is False

    nonstandard = check_citations("Claim 【1†source】.", valid_source_ids=[1])
    assert nonstandard["ok"] is False
    assert nonstandard["nonstandard_sources"] == [1]


def test_exact_no_information_answer_is_valid_without_citations() -> None:
    validation = check_citations(NO_INFORMATION, valid_source_ids=[])

    assert validation["ok"] is True
    assert validation["cited_sources"] == []
