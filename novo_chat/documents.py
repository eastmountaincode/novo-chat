"""Service-only, checksummed staging for normalized Novo notebook exports."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    PROTOCOL_VERSION,
    NotebookScope,
    PageDocument,
    canonical_json,
    document_pages_checksum,
    ingest_pages_checksum,
)


_STORAGE_KEY = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_STORAGE_QUOTA_BYTES = 50 * 1024 * 1024 * 1024


class DocumentStoreError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class FinalizedDocument:
    scope: NotebookScope
    document_checksum: str
    pages: tuple[PageDocument, ...]


def _scope_value(scope: NotebookScope) -> dict[str, Any]:
    return scope.model_dump(mode="json", by_alias=True)


def _scope_key(scope: NotebookScope) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(_scope_value(scope))).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored document data is invalid.") from exc
    if not isinstance(value, dict):
        raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored document data is invalid.")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class DocumentStore:
    """Stages immutable batches and promotes only a completely verified export."""

    def __init__(self, root: str | Path, *, max_bytes: int = DEFAULT_STORAGE_QUOTA_BYTES):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("document storage quota must be a positive integer")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.staging_root = self.root / "staging"
        self.final_root = self.root / "finalized"
        for directory in (self.root, self.staging_root, self.final_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass
        self._bytes_used = self._tree_size(self.root)

    def healthy(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.R_OK | os.W_OK | os.X_OK)

    def _staging_directory(self, scope: NotebookScope) -> Path:
        return self.staging_root / _scope_key(scope)

    def _final_path(self, scope: NotebookScope) -> Path:
        return self.final_root / _scope_key(scope) / "document.json"

    def stage_batch(
        self,
        *,
        scope: NotebookScope,
        batch_number: int,
        batch_checksum: str,
        pages: Sequence[PageDocument | Mapping[str, Any]],
    ) -> int:
        normalized = tuple(
            page if isinstance(page, PageDocument) else PageDocument.model_validate(page) for page in pages
        )
        if ingest_pages_checksum(normalized) != batch_checksum:
            raise DocumentStoreError("CHECKSUM_MISMATCH", "Ingest batch checksum does not match its content.")
        batch_payload = {
            "protocolVersion": PROTOCOL_VERSION,
            "scope": _scope_value(scope),
            "batchNumber": int(batch_number),
            "batchChecksum": batch_checksum,
            "pages": [page.model_dump(mode="json", by_alias=True) for page in normalized],
        }
        path = self._staging_directory(scope) / f"{int(batch_number):08d}.json"
        if path.exists():
            existing = _read_json(path)
            if canonical_json(existing) != canonical_json(batch_payload):
                raise DocumentStoreError(
                    "INGEST_BATCH_CONFLICT",
                    "A different batch already exists for this notebook revision.",
                )
            return len(normalized)
        self._ensure_capacity(len(canonical_json(batch_payload)))
        _atomic_json(path, batch_payload)
        self._bytes_used += path.stat().st_size
        return len(normalized)

    def finalize(
        self,
        *,
        scope: NotebookScope,
        document_checksum: str,
        page_count: int,
        batch_count: int,
    ) -> FinalizedDocument:
        staging_directory = self._staging_directory(scope)
        present_numbers: set[int] = set()
        if staging_directory.exists():
            for path in staging_directory.iterdir():
                if path.is_file() and path.suffix == ".json" and path.stem.isdigit():
                    present_numbers.add(int(path.stem))
        expected_numbers = set(range(int(batch_count)))
        if present_numbers != expected_numbers:
            raise DocumentStoreError(
                "INGEST_BATCHES_INCOMPLETE",
                "Ingest batches are missing, duplicated, or not contiguous.",
                retryable=True,
            )

        pages: list[PageDocument] = []
        for batch_number in range(int(batch_count)):
            payload = _read_json(staging_directory / f"{batch_number:08d}.json")
            try:
                stored_scope = NotebookScope.model_validate(payload.get("scope"))
                stored_number = int(payload.get("batchNumber"))
                stored_checksum = str(payload.get("batchChecksum") or "")
                stored_pages = tuple(PageDocument.model_validate(page) for page in payload.get("pages", []))
            except (TypeError, ValueError) as exc:
                raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored ingest batch is invalid.") from exc
            if stored_scope != scope or stored_number != batch_number:
                raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored ingest batch is invalid.")
            if ingest_pages_checksum(stored_pages) != stored_checksum:
                raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored ingest batch checksum is invalid.")
            pages.extend(stored_pages)

        page_ids = [page.page_id for page in pages]
        if len(page_ids) != len(set(page_ids)):
            raise DocumentStoreError("DUPLICATE_PAGE", "Notebook export contains a duplicate page ID.")
        if len(pages) != int(page_count):
            raise DocumentStoreError("PAGE_COUNT_MISMATCH", "Notebook export page count does not match.")
        actual_document_checksum = document_pages_checksum(pages)
        if actual_document_checksum != document_checksum:
            raise DocumentStoreError("DOCUMENT_CHECKSUM_MISMATCH", "Notebook export checksum does not match.")

        final_payload = {
            "protocolVersion": PROTOCOL_VERSION,
            "scope": _scope_value(scope),
            "documentChecksum": document_checksum,
            "pageCount": len(pages),
            "batchCount": int(batch_count),
            "pages": [page.model_dump(mode="json", by_alias=True) for page in pages],
        }
        final_path = self._final_path(scope)
        if final_path.exists():
            existing = _read_json(final_path)
            if str(existing.get("documentChecksum") or "") != document_checksum:
                raise DocumentStoreError(
                    "CONTENT_REVISION_CONFLICT",
                    "Stored content differs for the same declared content revision.",
                )
        else:
            self._ensure_capacity(len(canonical_json(final_payload)))
            _atomic_json(final_path, final_payload)
            self._bytes_used += final_path.stat().st_size
        # A finalized document is the canonical immutable copy. Keeping the
        # same normalized page bodies in staging would double sensitive-data
        # retention and let successful ingests grow without bound.
        if staging_directory.exists():
            staged_bytes = self._tree_size(staging_directory)
            shutil.rmtree(staging_directory, ignore_errors=False)
            self._bytes_used = max(0, self._bytes_used - staged_bytes)
        return FinalizedDocument(scope=scope, document_checksum=document_checksum, pages=tuple(pages))

    def load(self, scope: NotebookScope) -> FinalizedDocument:
        path = self._final_path(scope)
        if not path.exists():
            raise DocumentStoreError(
                "DOCUMENT_NOT_FINALIZED",
                "Notebook content has not been finalized for this exact revision.",
                retryable=True,
            )
        payload = _read_json(path)
        try:
            stored_scope = NotebookScope.model_validate(payload.get("scope"))
            pages = tuple(PageDocument.model_validate(page) for page in payload.get("pages", []))
            checksum = str(payload.get("documentChecksum") or "")
        except (TypeError, ValueError) as exc:
            raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored finalized document is invalid.") from exc
        if stored_scope != scope or checksum != document_pages_checksum(pages):
            raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored finalized document is invalid.")
        return FinalizedDocument(scope=scope, document_checksum=checksum, pages=pages)

    def prune(
        self,
        *,
        retained_finalized_scopes: Sequence[NotebookScope],
        retained_staging_scopes: Sequence[NotebookScope] = (),
        modified_before: float,
    ) -> dict[str, int]:
        """Remove expired derived documents outside the authoritative keep set.

        Callers obtain the keep set from the job/index database while the
        worker is idle. Only service-created SHA-256 directories are eligible;
        unexpected entries fail closed instead of widening a deletion target.
        """

        retained_finalized = {_scope_key(scope) for scope in retained_finalized_scopes}
        retained_staging = {_scope_key(scope) for scope in retained_staging_scopes}
        return {
            "staging": self._prune_root(
                self.staging_root,
                retained_keys=retained_staging,
                modified_before=modified_before,
            ),
            "finalized": self._prune_root(
                self.final_root,
                retained_keys=retained_finalized,
                modified_before=modified_before,
            ),
        }

    def _prune_root(self, root: Path, *, retained_keys: set[str], modified_before: float) -> int:
        removed = 0
        for candidate in root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir() or not _STORAGE_KEY.fullmatch(candidate.name):
                raise DocumentStoreError(
                    "DOCUMENT_STORE_CORRUPT",
                    "Stored document directory is invalid.",
                )
            if candidate.name in retained_keys or candidate.stat().st_mtime >= float(modified_before):
                continue
            shutil.rmtree(candidate)
            removed += 1
        if removed:
            self._bytes_used = self._tree_size(self.root)
        return removed

    def _ensure_capacity(self, additional_bytes: int) -> None:
        if additional_bytes < 0 or self._bytes_used + additional_bytes > self.max_bytes:
            raise DocumentStoreError(
                "STORAGE_QUOTA_EXCEEDED",
                "Normalized document storage reached its configured limit.",
            )

    @staticmethod
    def _tree_size(root: Path) -> int:
        total = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise DocumentStoreError("DOCUMENT_STORE_CORRUPT", "Stored document path is invalid.")
            if path.is_file():
                total += path.stat().st_size
        return total


__all__ = ["DEFAULT_STORAGE_QUOTA_BYTES", "DocumentStore", "DocumentStoreError", "FinalizedDocument"]
