from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from novo_chat.compute import ComputeError, IndexRepository
from novo_chat.documents import DocumentStore, DocumentStoreError, FinalizedDocument
from novo_chat.jobs import JobStore
from novo_chat.protocol import (
    JobOperation,
    NotebookScope,
    PageDocument,
    document_pages_checksum,
    ingest_pages_checksum,
)


class RetentionBackend:
    def embed_documents(self, texts, *, scope):
        del scope
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text):
        del text
        return np.ones(2, dtype=np.float32)

    def generate(self, question, hits, *, model, max_sources):
        del question, hits, model, max_sources
        return "The provided documents do not contain this information."


def scope(notebook: str, revision: str) -> NotebookScope:
    return NotebookScope(
        notebook_id=notebook,
        content_revision=revision,
        index_schema_version="index-v1",
    )


def page(page_id: str) -> PageDocument:
    return PageDocument(
        page_id=page_id,
        title=f"Page {page_id}",
        text="normalized notebook text",
        source_url=f"/?page={page_id}",
    )


def finalize(store: DocumentStore, item: NotebookScope, item_page: PageDocument) -> None:
    store.stage_batch(
        scope=item,
        batch_number=0,
        batch_checksum=ingest_pages_checksum((item_page,)),
        pages=(item_page,),
    )
    store.finalize(
        scope=item,
        document_checksum=document_pages_checksum((item_page,)),
        page_count=1,
        batch_count=1,
    )


def test_successful_finalize_removes_duplicate_staging_bodies(tmp_path: Path) -> None:
    documents = DocumentStore(tmp_path / "documents")
    item = scope("notebook-a", "revision-a")
    item_page = page("page-a")

    documents.stage_batch(
        scope=item,
        batch_number=0,
        batch_checksum=ingest_pages_checksum((item_page,)),
        pages=(item_page,),
    )
    assert any(documents.staging_root.iterdir())
    documents.finalize(
        scope=item,
        document_checksum=document_pages_checksum((item_page,)),
        page_count=1,
        batch_count=1,
    )

    assert list(documents.staging_root.iterdir()) == []
    assert documents.load(item).pages == (item_page,)


def test_document_prune_preserves_only_retained_finalized_scope(tmp_path: Path) -> None:
    documents = DocumentStore(tmp_path / "documents")
    retained = scope("notebook-retained", "revision-current")
    obsolete = scope("notebook-obsolete", "revision-old")
    abandoned = scope("notebook-abandoned", "revision-partial")
    finalize(documents, retained, page("page-retained"))
    finalize(documents, obsolete, page("page-obsolete"))
    documents.stage_batch(
        scope=abandoned,
        batch_number=0,
        batch_checksum=ingest_pages_checksum((page("page-abandoned"),)),
        pages=(page("page-abandoned"),),
    )
    old_time = 1_600_000_000
    for directory in (
        documents._final_path(retained).parent,
        documents._final_path(obsolete).parent,
        documents._staging_directory(abandoned),
    ):
        os.utime(directory, (old_time, old_time))

    removed = documents.prune(
        retained_finalized_scopes=(retained,),
        modified_before=old_time + 1,
    )

    assert removed == {"staging": 1, "finalized": 1}
    assert documents.load(retained).scope == retained
    assert not documents._final_path(obsolete).exists()
    assert not documents._staging_directory(abandoned).exists()


def test_index_prune_preserves_active_artifact_and_removes_old_candidates(tmp_path: Path) -> None:
    indexes = IndexRepository(tmp_path / "indexes", RetentionBackend())
    retained_scope = scope("notebook-a", "revision-current")
    obsolete_scope = scope("notebook-a", "revision-old")
    retained_candidate = indexes.build_candidate(
        FinalizedDocument(retained_scope, document_pages_checksum(()), ()),
        operation_id="operation-retained",
    )
    retained = indexes.commit_candidate(retained_candidate)
    obsolete_candidate = indexes.build_candidate(
        FinalizedDocument(obsolete_scope, document_pages_checksum(()), ()),
        operation_id="operation-obsolete",
    )
    obsolete = indexes.commit_candidate(obsolete_candidate)
    abandoned = indexes.build_candidate(
        FinalizedDocument(scope("notebook-b", "revision-partial"), document_pages_checksum(()), ()),
        operation_id="operation-abandoned",
    )
    old_time = 1_600_000_000
    for directory in (
        indexes.artifact_root / retained.artifact_id,
        indexes.artifact_root / obsolete.artifact_id,
        abandoned.temporary_path,
    ):
        os.utime(directory, (old_time, old_time))

    removed = indexes.prune(
        retained_artifact_ids=(retained.artifact_id,),
        modified_before=old_time + 1,
    )

    assert removed == {"candidates": 1, "artifacts": 1}
    assert (indexes.artifact_root / retained.artifact_id).is_dir()
    assert not (indexes.artifact_root / obsolete.artifact_id).exists()
    assert not abandoned.temporary_path.exists()


def test_storage_snapshot_expires_old_indexes_only_while_idle(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "worker.sqlite3")
    old_scope = scope("notebook-old", "revision-old")
    fresh_scope = scope("notebook-fresh", "revision-current")
    store.activate_index(old_scope, environment="staging", artifact_id="a" * 64, now=100)
    store.activate_index(fresh_scope, environment="staging", artifact_id="b" * 64, now=300)

    idle = store.storage_retention_snapshot(environment="staging", activated_before=200)
    assert idle.idle is True
    assert idle.expired_indexes == 1
    assert idle.active_scopes == (fresh_scope,)
    assert idle.active_artifact_ids == ("b" * 64,)

    store.submit(
        environment="staging",
        actor_user_id="user-a",
        idempotency_key=f"idem-{uuid4()}",
        request_id=str(uuid4()),
        operation=JobOperation.INDEX_REBUILD,
        scope=(fresh_scope,),
        payload={"force": False},
        now=400,
    )
    busy = store.storage_retention_snapshot(environment="staging", activated_before=500)
    assert busy.idle is False
    assert busy.expired_indexes == 0
    assert busy.active_scopes == (fresh_scope,)


def test_document_and_index_stores_enforce_configured_byte_quotas(tmp_path: Path) -> None:
    item = scope("notebook-quota", "revision-quota")
    item_page = page("page-quota")
    documents = DocumentStore(tmp_path / "documents", max_bytes=64)
    try:
        documents.stage_batch(
            scope=item,
            batch_number=0,
            batch_checksum=ingest_pages_checksum((item_page,)),
            pages=(item_page,),
        )
    except DocumentStoreError as exc:
        assert exc.code == "STORAGE_QUOTA_EXCEEDED"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("document quota was not enforced")

    indexes = IndexRepository(tmp_path / "indexes", RetentionBackend(), max_bytes=64)
    try:
        indexes.build_candidate(
            FinalizedDocument(item, document_pages_checksum(()), ()),
            operation_id="operation-quota",
        )
    except ComputeError as exc:
        assert exc.code == "STORAGE_QUOTA_EXCEEDED"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("index quota was not enforced")
