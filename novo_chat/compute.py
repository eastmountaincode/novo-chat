"""Exact-version index artifacts and hybrid retrieval for the worker.

This module consumes only normalized PageDocument objects. It deliberately has
no Novo database, authentication, container, or network imports. Production
model I/O is injected through ``ComputeBackend``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .documents import FinalizedDocument
from .jobs import ActiveIndex
from .protocol import (
    Citation,
    GenerationResult,
    JobProgressDetail,
    MAX_CITATION_EXCERPT_CHARS,
    NotebookScope,
    PageDocument,
    QueryJobResult,
    QueryProgressStage,
    QueryStrategy,
    RetrievalPlan,
    canonical_json,
)


_ARTIFACT_ID = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_DIRECTORY = re.compile(r"^[0-9a-f]{64}\.[A-Za-z0-9_-]+$")
_SEMANTIC_EXPANSION_RRF_WEIGHT = 0.85
_LEXICAL_EXPANSION_RRF_WEIGHT = 0.9
DEFAULT_STORAGE_QUOTA_BYTES = 50 * 1024 * 1024 * 1024


class ComputeError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ComputeBackend(Protocol):
    def embed_documents(self, texts: Sequence[str], *, scope: NotebookScope) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...

    def plan_query(
        self,
        question: str,
        retrieval_question: str,
        *,
        model: str,
    ) -> RetrievalPlan:
        ...

    def generate(
        self,
        question: str,
        hits: Sequence[Mapping[str, Any]],
        *,
        model: str,
        max_sources: int,
    ) -> str | GenerationResult:
        ...


@dataclass(frozen=True)
class IndexCandidate:
    scope: NotebookScope
    artifact_id: str
    temporary_path: Path
    chunk_count: int


@dataclass(frozen=True)
class LoadedIndexArtifact:
    scope: NotebookScope
    artifact_id: str
    chunks: tuple[dict[str, Any], ...]
    vectors: np.ndarray


@dataclass(frozen=True)
class IndexArtifactStatus:
    exact_ready: bool
    activated_at: str | None = None
    chunk_count: int | None = None


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _normalize_vectors(vectors: np.ndarray, expected_count: int) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != expected_count:
        raise ComputeError("EMBEDDING_INVALID", "Embedding backend returned an invalid matrix.", retryable=True)
    if expected_count and value.shape[1] < 1:
        raise ComputeError("EMBEDDING_INVALID", "Embedding backend returned an invalid matrix.", retryable=True)
    if not np.all(np.isfinite(value)):
        raise ComputeError("EMBEDDING_INVALID", "Embedding backend returned invalid values.", retryable=True)
    if expected_count:
        norms = np.linalg.norm(value, axis=1, keepdims=True)
        value = value / np.clip(norms, 1e-10, None)
    return value


def _chunks_for_pages(scope: NotebookScope, pages: Sequence[PageDocument]) -> list[dict[str, Any]]:
    from .rag_core import chunk_text

    chunks: list[dict[str, Any]] = []
    for page in pages:
        attachment_names = [attachment.name for attachment in page.attachments]
        metadata_lines = [
            f"Notebook ID: {scope.notebook_id}",
            f"Title: {page.title}",
            f"Page ID: {page.page_id}",
            f"Created: {page.created_at or ''}",
            f"Updated: {page.updated_at or ''}",
        ]
        if page.tags:
            metadata_lines.append("Tags: " + ", ".join(page.tags))
        if attachment_names:
            metadata_lines.append("Attachments: " + "; ".join(attachment_names))
        metadata_text = "\n".join(metadata_lines)
        pieces = chunk_text(page.text.strip()) or [""]
        for chunk_number, text in enumerate(pieces):
            chunks.append(
                {
                    "file": f"{page.page_id}.md",
                    "title": page.title,
                    "chunk_idx": chunk_number,
                    "text": text,
                    "indexed_text": f"{metadata_text}\n\n{text}".strip(),
                    "metadata": {
                        "created": page.created_at or "",
                        "updated": page.updated_at or "",
                    },
                    "page_id": page.page_id,
                    "notebook_id": scope.notebook_id,
                    "notebook": scope.notebook_id,
                    "updated": page.updated_at or "",
                    "tags": list(page.tags),
                    "attachments": attachment_names,
                    "source_url": page.source_url,
                    "novo_url": page.source_url,
                }
            )
    return chunks


class IndexRepository:
    def __init__(
        self,
        root: str | Path,
        backend: ComputeBackend,
        *,
        max_bytes: int = DEFAULT_STORAGE_QUOTA_BYTES,
    ):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("index storage quota must be a positive integer")
        self.root = Path(root)
        self.candidate_root = self.root / "candidates"
        self.artifact_root = self.root / "artifacts"
        self.backend = backend
        self.max_bytes = max_bytes
        for directory in (self.root, self.candidate_root, self.artifact_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass
        self._bytes_used = self._tree_size(self.root)

    def healthy(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.R_OK | os.W_OK | os.X_OK)

    def build_candidate(self, document: FinalizedDocument, *, operation_id: str) -> IndexCandidate:
        import hashlib
        import uuid

        chunks = _chunks_for_pages(document.scope, document.pages)
        vectors = _normalize_vectors(
            self.backend.embed_documents(
                [str(chunk["indexed_text"]) for chunk in chunks],
                scope=document.scope,
            ),
            len(chunks),
        )
        artifact_id = hashlib.sha256(
            canonical_json(
                {
                    "operationId": operation_id,
                    "scope": document.scope.model_dump(mode="json", by_alias=True),
                    "documentChecksum": document.document_checksum,
                    "nonce": uuid.uuid4().hex,
                }
            )
        ).hexdigest()
        estimated_bytes = len(canonical_json(chunks)) + int(vectors.nbytes) + 16 * 1024
        self._ensure_capacity(estimated_bytes)
        candidate_path = Path(tempfile.mkdtemp(prefix=f"{artifact_id}.", dir=str(self.candidate_root)))
        try:
            os.chmod(candidate_path, 0o700)
            _atomic_json(candidate_path / "chunks.json", chunks)
            vector_path = candidate_path / "vectors.npy"
            with vector_path.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(vector_path, 0o600)
            _atomic_json(
                candidate_path / "manifest.json",
                {
                    "version": 1,
                    "artifactId": artifact_id,
                    "scope": document.scope.model_dump(mode="json", by_alias=True),
                    "documentChecksum": document.document_checksum,
                    "chunkCount": len(chunks),
                    "vectorDimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
                    "createdAt": int(time.time()),
                },
            )
            self._load_path(candidate_path, expected_scope=document.scope, expected_artifact_id=artifact_id)
            self._bytes_used += self._tree_size(candidate_path)
            return IndexCandidate(
                scope=document.scope,
                artifact_id=artifact_id,
                temporary_path=candidate_path,
                chunk_count=len(chunks),
            )
        except Exception:
            shutil.rmtree(candidate_path, ignore_errors=True)
            raise

    def commit_candidate(self, candidate: IndexCandidate) -> ActiveIndex:
        target = self.artifact_root / candidate.artifact_id
        if target.exists():
            candidate_bytes = self._tree_size(candidate.temporary_path)
            shutil.rmtree(candidate.temporary_path, ignore_errors=True)
            self._bytes_used = max(0, self._bytes_used - candidate_bytes)
        else:
            os.replace(candidate.temporary_path, target)
        self._load_path(target, expected_scope=candidate.scope, expected_artifact_id=candidate.artifact_id)
        return ActiveIndex(scope=candidate.scope, artifact_id=candidate.artifact_id, activated_at="")

    def discard_candidate(self, candidate: IndexCandidate) -> None:
        candidate_bytes = self._tree_size(candidate.temporary_path) if candidate.temporary_path.exists() else 0
        shutil.rmtree(candidate.temporary_path, ignore_errors=True)
        self._bytes_used = max(0, self._bytes_used - candidate_bytes)

    def load_active(self, active: ActiveIndex) -> LoadedIndexArtifact:
        if not active.artifact_id or not _ARTIFACT_ID.fullmatch(active.artifact_id):
            raise ComputeError("INDEX_NOT_READY", "An exact requested notebook index is not ready.", retryable=True)
        return self._load_path(
            self.artifact_root / active.artifact_id,
            expected_scope=active.scope,
            expected_artifact_id=active.artifact_id,
        )

    def artifact_ready(self, active: ActiveIndex | None) -> bool:
        return self.artifact_status(active).exact_ready

    def artifact_status(self, active: ActiveIndex | None) -> IndexArtifactStatus:
        if active is None:
            return IndexArtifactStatus(exact_ready=False)
        try:
            artifact = self.load_active(active)
        except ComputeError:
            return IndexArtifactStatus(exact_ready=False)
        activated_at = str(active.activated_at).strip()
        if not activated_at or len(activated_at) > 64:
            activated_at = None
        return IndexArtifactStatus(
            exact_ready=True,
            activated_at=activated_at,
            chunk_count=len(artifact.chunks),
        )

    def prune(self, *, retained_artifact_ids: Sequence[str], modified_before: float) -> dict[str, int]:
        """Remove expired candidates and non-active immutable artifacts."""

        # Legacy catalog rows may have an empty artifact ID. They cannot name a
        # filesystem artifact and therefore do not broaden the keep set.
        retained = {artifact_id for artifact_id in retained_artifact_ids if _ARTIFACT_ID.fullmatch(artifact_id)}
        result = {
            "candidates": self._prune_root(
                self.candidate_root,
                retained_ids=set(),
                modified_before=modified_before,
                candidate_names=True,
            ),
            "artifacts": self._prune_root(
                self.artifact_root,
                retained_ids=retained,
                modified_before=modified_before,
                candidate_names=False,
            ),
        }
        if result["candidates"] or result["artifacts"]:
            self._bytes_used = self._tree_size(self.root)
        return result

    @staticmethod
    def _prune_root(
        root: Path,
        *,
        retained_ids: set[str],
        modified_before: float,
        candidate_names: bool,
    ) -> int:
        removed = 0
        expected_name = _CANDIDATE_DIRECTORY if candidate_names else _ARTIFACT_ID
        for candidate in root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir() or not expected_name.fullmatch(candidate.name):
                raise ComputeError("INDEX_STORE_CORRUPT", "Stored index directory is invalid.")
            artifact_id = candidate.name.split(".", 1)[0]
            if artifact_id in retained_ids or candidate.stat().st_mtime >= float(modified_before):
                continue
            shutil.rmtree(candidate)
            removed += 1
        return removed

    def _ensure_capacity(self, additional_bytes: int) -> None:
        if additional_bytes < 0 or self._bytes_used + additional_bytes > self.max_bytes:
            raise ComputeError(
                "STORAGE_QUOTA_EXCEEDED",
                "Index storage reached its configured limit.",
            )

    @staticmethod
    def _tree_size(root: Path) -> int:
        total = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ComputeError("INDEX_STORE_CORRUPT", "Stored index path is invalid.")
            if path.is_file():
                total += path.stat().st_size
        return total

    def _load_path(
        self,
        path: Path,
        *,
        expected_scope: NotebookScope,
        expected_artifact_id: str,
    ) -> LoadedIndexArtifact:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            chunks_value = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
            with (path / "vectors.npy").open("rb") as handle:
                vectors = np.load(handle, allow_pickle=False)
            scope = NotebookScope.model_validate(manifest.get("scope"))
            artifact_id = str(manifest.get("artifactId") or "")
            chunks = tuple(dict(chunk) for chunk in chunks_value)
        except Exception as exc:
            raise ComputeError("INDEX_CORRUPT", "Stored index artifact is invalid.") from exc
        if scope != expected_scope or artifact_id != expected_artifact_id:
            raise ComputeError("INDEX_CORRUPT", "Stored index artifact is invalid.")
        vectors = _normalize_vectors(vectors, len(chunks))
        if int(manifest.get("chunkCount", -1)) != len(chunks):
            raise ComputeError("INDEX_CORRUPT", "Stored index artifact is invalid.")
        if any(str(chunk.get("notebook_id") or "") != scope.notebook_id for chunk in chunks):
            raise ComputeError("INDEX_CORRUPT", "Stored index exceeded its notebook scope.")
        return LoadedIndexArtifact(scope=scope, artifact_id=artifact_id, chunks=chunks, vectors=vectors)

    def query(
        self,
        artifacts: Sequence[LoadedIndexArtifact],
        *,
        question: str,
        retrieval_question: str,
        model: str,
        strategy: QueryStrategy,
        max_sources: int,
        retrieval_top_k: int = 16,
        progress_callback: Callable[[JobProgressDetail], None] | None = None,
    ) -> QueryJobResult:
        for label, value, upper_bound in (
            ("maxSources", max_sources, 100),
            ("retrievalTopK", retrieval_top_k, 32),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= upper_bound:
                raise ComputeError(
                    "QUERY_OPTIONS_INVALID",
                    f"{label} must be an integer between 1 and {upper_bound}.",
                )
        allowed_notebooks = {artifact.scope.notebook_id for artifact in artifacts}
        chunks: list[dict[str, Any]] = []
        vector_sets: list[np.ndarray] = []
        dimension: int | None = None
        for artifact in artifacts:
            if any(str(chunk.get("notebook_id") or "") not in allowed_notebooks for chunk in artifact.chunks):
                raise ComputeError("INDEX_SCOPE_VIOLATION", "Stored index exceeded requested notebook scope.")
            if artifact.vectors.shape[0]:
                artifact_dimension = int(artifact.vectors.shape[1])
                if dimension is not None and artifact_dimension != dimension:
                    raise ComputeError("INDEX_INCOMPATIBLE", "Requested indexes use incompatible embeddings.")
                dimension = artifact_dimension
                vector_sets.append(artifact.vectors)
            chunks.extend(dict(chunk) for chunk in artifact.chunks)

        plan = self._query_plan(
            question=question,
            retrieval_question=retrieval_question,
            model=model,
            use_model=bool(chunks),
        )
        if progress_callback is not None:
            progress_callback(
                JobProgressDetail(
                    stage=QueryProgressStage.SEARCHING,
                    retrieval_plan=plan,
                )
            )
        if not chunks:
            if progress_callback is not None:
                progress_callback(
                    JobProgressDetail(
                        stage=QueryProgressStage.ANSWERING,
                        retrieval_plan=plan,
                        retrieved_count=0,
                    )
                )
            generation = self._generation_result(
                self.backend.generate(question, (), model=model, max_sources=max_sources)
            )
            return QueryJobResult(
                answer=self._validated_answer(generation.answer, ()),
                model=model,
                citations=(),
                timings=generation.timings,
                retrieval_plan=plan,
            )
        vectors = np.vstack(vector_sets).astype(np.float32, copy=False)
        ranked_hits = self._retrieve(
            chunks,
            vectors,
            question,
            plan,
            strategy=strategy,
            count=retrieval_top_k,
        )
        prompt_hits = ranked_hits[:max_sources]
        if progress_callback is not None:
            progress_callback(
                JobProgressDetail(
                    stage=QueryProgressStage.ANSWERING,
                    retrieval_plan=plan,
                    retrieved_count=len(prompt_hits),
                )
            )
        generation = self._generation_result(
            self.backend.generate(question, prompt_hits, model=model, max_sources=max_sources)
        )
        answer = self._validated_answer(generation.answer, prompt_hits)
        citations = tuple(
            Citation(
                source_idx=int(hit["source_idx"]),
                file=str(hit.get("file") or ""),
                notebook_id=str(hit["notebook_id"]),
                page_id=str(hit["page_id"]),
                chunk_idx=int(hit.get("chunk_idx") or 0),
                score=float(hit["score"]),
                bm25=float(hit["bm25"]),
                dense=float(hit["dense"]),
                excerpt=str(hit.get("text") or "")[:MAX_CITATION_EXCERPT_CHARS],
                source_url=str(hit["source_url"]),
                title=str(hit.get("title") or ""),
                used_in_context=position <= len(prompt_hits),
            )
            for position, hit in enumerate(ranked_hits, start=1)
        )
        return QueryJobResult(
            answer=answer,
            model=model,
            citations=citations,
            timings=generation.timings,
            retrieval_plan=plan,
        )

    def _query_plan(
        self,
        *,
        question: str,
        retrieval_question: str,
        model: str,
        use_model: bool,
    ) -> RetrievalPlan:
        from .rag_core import fallback_retrieval_plan

        fallback = fallback_retrieval_plan(question, retrieval_question)
        planner = getattr(self.backend, "plan_query", None)
        if not use_model or not callable(planner):
            return fallback
        try:
            planned = RetrievalPlan.model_validate(
                planner(question, retrieval_question, model=model)
            )
            return RetrievalPlan(
                original_question=question,
                semantic_query=planned.semantic_query,
                bm25_terms=planned.bm25_terms,
                mode=planned.mode,
            )
        except Exception:
            return fallback

    @staticmethod
    def _generation_result(value: Any) -> GenerationResult:
        if isinstance(value, str):
            return GenerationResult(answer=value)
        try:
            return GenerationResult.model_validate(value)
        except (TypeError, ValueError) as exc:
            raise ComputeError(
                "GENERATION_RESULT_INVALID",
                "The configured generation service returned an invalid result.",
                retryable=True,
            ) from exc

    @staticmethod
    def _validated_answer(answer: str, hits: Sequence[Mapping[str, Any]]) -> str:
        from .rag_core import check_citations, normalize_citation_glyphs

        normalized, _normalized_sources = normalize_citation_glyphs(answer)
        if not hits and not normalized.strip().startswith(
            "The provided documents do not contain this information."
        ):
            raise ComputeError(
                "CITATION_VALIDATION_FAILED",
                "Generated answer did not contain valid source citations.",
                retryable=True,
            )
        source_ids = [int(hit["source_idx"]) for hit in hits]
        validation = check_citations(normalized, valid_source_ids=source_ids)
        if not validation.get("ok"):
            raise ComputeError(
                "CITATION_VALIDATION_FAILED",
                "Generated answer did not contain valid source citations.",
                retryable=True,
            )
        return normalized

    def _retrieve(
        self,
        chunks: Sequence[dict[str, Any]],
        vectors: np.ndarray,
        question: str,
        plan: RetrievalPlan,
        *,
        strategy: QueryStrategy,
        count: int,
    ) -> list[dict[str, Any]]:
        from .rag_core import diversify_by_page, meaningful_search_tokens, tokenize

        pool = min(240, len(chunks))
        original_fused: dict[int, float] = {}
        semantic_expansion: dict[int, float] = {}
        lexical_expansion: dict[int, float] = {}
        dense_scores = np.zeros(len(chunks), dtype=np.float32)
        lexical_scores = np.zeros(len(chunks), dtype=np.float32)

        if strategy in {QueryStrategy.HYBRID, QueryStrategy.SEMANTIC}:
            dense_queries: list[tuple[str, bool]] = []
            seen_queries: set[str] = set()
            for candidate, is_expansion in (
                (question, False),
                (plan.semantic_query, True),
            ):
                normalized = candidate.strip()
                key = " ".join(tokenize(normalized)) or normalized.casefold()
                if not normalized or key in seen_queries:
                    continue
                seen_queries.add(key)
                dense_queries.append((normalized, is_expansion))
            dense_score_sets: list[np.ndarray] = []
            for dense_query, is_expansion in dense_queries:
                query_vector = np.asarray(
                    self.backend.embed_query(dense_query),
                    dtype=np.float32,
                ).reshape(-1)
                if not np.all(np.isfinite(query_vector)) or vectors.shape[1] != query_vector.shape[0]:
                    raise ComputeError(
                        "QUERY_EMBEDDING_INVALID",
                        "Query embedding is incompatible.",
                        retryable=True,
                    )
                query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-10)
                scores = vectors @ query_vector
                dense_score_sets.append(scores)
                dense_top = np.lexsort((np.arange(len(scores)), -scores))[:pool]
                target = semantic_expansion if is_expansion else original_fused
                weight = _SEMANTIC_EXPANSION_RRF_WEIGHT if is_expansion else 1.0
                for rank, index in enumerate(dense_top):
                    index = int(index)
                    target[index] = target.get(index, 0.0) + weight / (60 + rank)
            if dense_score_sets:
                dense_scores = np.max(np.vstack(dense_score_sets), axis=0)

        if strategy in {QueryStrategy.HYBRID, QueryStrategy.LEXICAL}:
            tokenized_chunks = [
                tokenize(str(chunk.get("indexed_text") or "")) for chunk in chunks
            ]
            bm25 = BM25Okapi(tokenized_chunks)
            original_tokens: list[str] = []
            seen_original_tokens: set[str] = set()
            for token in meaningful_search_tokens(question) or tokenize(question):
                if token in seen_original_tokens:
                    continue
                seen_original_tokens.add(token)
                original_tokens.append(token)
            expansion_tokens: list[str] = []
            seen_expansion_tokens: set[str] = set()
            for term in plan.bm25_terms:
                for token in meaningful_search_tokens(term):
                    if token in seen_original_tokens or token in seen_expansion_tokens:
                        continue
                    seen_expansion_tokens.add(token)
                    expansion_tokens.append(token)

            lexical_score_sets: list[np.ndarray] = []
            for lexical_tokens, lexical_weight, target, is_expansion in (
                (original_tokens, 1.0, original_fused, False),
                (
                    expansion_tokens,
                    _LEXICAL_EXPANSION_RRF_WEIGHT,
                    lexical_expansion,
                    True,
                ),
            ):
                if not lexical_tokens:
                    continue
                scores = np.asarray(bm25.get_scores(lexical_tokens), dtype=np.float32)
                lexical_score_sets.append(lexical_weight * scores)
                lexical_token_set = set(lexical_tokens)
                overlap_counts = np.asarray(
                    [
                        len(lexical_token_set.intersection(chunk_tokens))
                        for chunk_tokens in tokenized_chunks
                    ],
                    dtype=np.float32,
                )
                matched_indices = np.flatnonzero(overlap_counts > 0)
                if not len(matched_indices):
                    continue
                lexical_max = float(np.max(scores[matched_indices]))
                ranking_scores = (
                    scores[matched_indices]
                    if lexical_max > 0
                    else overlap_counts[matched_indices]
                )
                lexical_order = np.lexsort((matched_indices, -ranking_scores))[:pool]
                lexical_top = matched_indices[lexical_order]
                for rank, index in enumerate(lexical_top):
                    index = int(index)
                    target[index] = target.get(index, 0.0) + lexical_weight / (60 + rank)
                if strategy is QueryStrategy.HYBRID and not is_expansion:
                    normalized_bonus = (
                        0.008 * np.maximum(scores, 0.0) / lexical_max
                        if lexical_max > 0
                        else 0.008 * overlap_counts / float(np.max(overlap_counts))
                    )
                    for index in np.flatnonzero(normalized_bonus > 0):
                        item = int(index)
                        target[item] = target.get(item, 0.0) + float(
                            normalized_bonus[item]
                        )
            if lexical_score_sets:
                lexical_scores = np.maximum(
                    np.max(np.vstack(lexical_score_sets), axis=0),
                    0.0,
                )

        fused = dict(original_fused)
        for index in semantic_expansion.keys() | lexical_expansion.keys():
            # Dense and lexical expansions are two views of the same untrusted
            # planner hypothesis, so they may not reinforce one another. Either
            # can fill a retrieval gap without reducing original-query evidence.
            expansion_score = max(
                semantic_expansion.get(index, 0.0),
                lexical_expansion.get(index, 0.0),
            )
            fused[index] = max(fused.get(index, 0.0), expansion_score)

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        selected = diversify_by_page(ranked, list(chunks), min(int(count), len(chunks)), max_per_page=2)
        hits: list[dict[str, Any]] = []
        for source_index, (chunk_index, score) in enumerate(selected, start=1):
            hit = dict(chunks[chunk_index])
            hit["source_idx"] = source_index
            hit["score"] = float(score)
            hit["dense"] = float(dense_scores[chunk_index])
            hit["bm25"] = float(lexical_scores[chunk_index])
            hits.append(hit)
        return hits


__all__ = [
    "ComputeBackend",
    "ComputeError",
    "IndexArtifactStatus",
    "IndexCandidate",
    "IndexRepository",
    "LoadedIndexArtifact",
]
