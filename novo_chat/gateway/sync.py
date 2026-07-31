from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from novo_chat.protocol import PageDocument, canonical_json, ingest_pages_checksum, new_request_id

from .config import GatewaySettings
from .novo_client import NovoIntegrationClient
from .worker_client import GatewayWorkerClient, WorkerRejected, unwrap_job


class SynchronizationError(Exception):
    """Safe browser-facing failure while preparing an exact notebook index."""


class SynchronizationTimeout(SynchronizationError):
    pass


class ExportLimitExceeded(SynchronizationError):
    pass


@dataclass(frozen=True, slots=True)
class ExportSummary:
    notebook_id: str
    page_count: int
    batch_count: int
    document_checksum: str


class CanonicalDocumentChecksum:
    """Incrementally hash the canonical JSON array of validated page documents."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self._hash.update(b"[")
        self._first = True

    def add(self, pages: Iterable[PageDocument]) -> None:
        for page in pages:
            if not self._first:
                self._hash.update(b",")
            self._hash.update(canonical_json(page))
            self._first = False

    def hexdigest(self) -> str:
        digest = self._hash.copy()
        digest.update(b"]")
        return f"sha256:{digest.hexdigest()}"


class GatewaySynchronizer:
    def __init__(
        self,
        *,
        novo_client: NovoIntegrationClient | Any,
        worker_client: GatewayWorkerClient | Any,
        settings: GatewaySettings,
    ) -> None:
        self.novo_client = novo_client
        self.worker_client = worker_client
        self.settings = settings

    async def prepare_documents_for_rebuild(
        self,
        *,
        session_value: str,
        actor_user_id: str,
        scope: list[dict[str, str]],
    ) -> None:
        """Ingest exact documents only where the worker lacks an exact index."""

        try:
            async with asyncio.timeout(self.settings.worker_job_timeout_s):
                missing = await self._missing_scope(actor_user_id, scope)
                for entry in missing:
                    await self._export_and_ingest(session_value, actor_user_id, entry)
        except TimeoutError as exc:
            raise SynchronizationTimeout("Notebook preparation timed out. Try again.") from exc

    async def ensure_indexes_ready(
        self,
        *,
        session_value: str,
        actor_user_id: str,
        scope: list[dict[str, str]],
    ) -> None:
        """Synchronize, rebuild, and verify every exact scope entry."""

        try:
            async with asyncio.timeout(self.settings.worker_job_timeout_s):
                missing = await self._missing_scope(actor_user_id, scope)
                if not missing:
                    return
                for entry in missing:
                    await self._export_and_ingest(session_value, actor_user_id, entry)
                rebuild = await self.worker_client.submit(
                    operation="index_rebuild",
                    request_id=new_request_id(),
                    idempotency_key=_idempotency_key("auto-rebuild", missing),
                    actor_user_id=actor_user_id,
                    scope=missing,
                    payload={"force": False},
                )
                await self._wait_for_job(rebuild, expected_operation="index_rebuild")
                still_missing = await self._missing_scope(actor_user_id, scope)
                if still_missing:
                    raise SynchronizationError("The compute worker did not activate the requested notebook index")
        except TimeoutError as exc:
            raise SynchronizationTimeout("Notebook preparation timed out. Try again.") from exc

    async def _missing_scope(
        self,
        actor_user_id: str,
        scope: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        response = await self.worker_client.index_status(
            request_id=new_request_id(),
            actor_user_id=actor_user_id,
            scope=scope,
        )
        rows = response.get("indexes")
        if not isinstance(rows, list) or len(rows) != len(scope):
            raise WorkerRejected("Worker returned an invalid index-status response")
        by_key: dict[tuple[str, str, str], bool] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise WorkerRejected("Worker returned an invalid index-status response")
            key = (
                str(row.get("notebookId") or ""),
                str(row.get("contentRevision") or ""),
                str(row.get("indexSchemaVersion") or ""),
            )
            if key in by_key:
                raise WorkerRejected("Worker returned duplicate index-status entries")
            exact_ready = row.get("exactReady")
            if not isinstance(exact_ready, bool):
                raise WorkerRejected("Worker returned an invalid index-status readiness value")
            by_key[key] = exact_ready
        missing: list[dict[str, str]] = []
        for entry in scope:
            key = (entry["notebookId"], entry["contentRevision"], entry["indexSchemaVersion"])
            if key not in by_key:
                raise WorkerRejected("Worker omitted an index-status entry")
            if not by_key[key]:
                missing.append(entry)
        return missing

    async def _export_and_ingest(
        self,
        session_value: str,
        actor_user_id: str,
        scope: dict[str, str],
    ) -> ExportSummary:
        notebook_id = scope["notebookId"]
        revision = scope["contentRevision"]
        cursor: str | None = None
        seen_cursors: set[str] = set()
        seen_page_ids: set[str] = set()
        page_count = 0
        batch_count = 0
        page_requests = 0
        checksum = CanonicalDocumentChecksum()

        while True:
            page_requests += 1
            if page_requests > self.settings.novo_export_max_batches:
                raise ExportLimitExceeded("Novo page export exceeded its pagination limit")
            exported = await self.novo_client.export_pages(
                session_value,
                notebook_id=notebook_id,
                content_revision=revision,
                cursor=cursor,
                limit=self.settings.novo_export_page_limit,
            )
            try:
                documents = [PageDocument.model_validate(page.worker_document()) for page in exported.pages]
            except ValueError as exc:
                raise SynchronizationError("Novo page export contained unsupported page content") from exc
            page_ids = [page.page_id for page in documents]
            if len(page_ids) != len(set(page_ids)) or any(page_id in seen_page_ids for page_id in page_ids):
                raise SynchronizationError("Novo page export returned a duplicate page")
            seen_page_ids.update(page_ids)
            if page_count + len(documents) > self.settings.novo_export_max_pages:
                raise ExportLimitExceeded("Novo page export exceeded its configured page limit")
            page_count += len(documents)
            checksum.add(documents)

            pages = [page.model_dump(mode="json", by_alias=True) for page in documents]
            for page_batch in _partition_page_batches(
                pages,
                max_canonical_bytes=self.settings.ingest_batch_max_bytes,
            ):
                batch_checksum = _page_batch_checksum(page_batch)
                submitted = await self.worker_client.ingest_batch(
                    request_id=new_request_id(),
                    idempotency_key=_idempotency_key(
                        "ingest-batch",
                        [scope],
                        extra=f"{batch_count}:{batch_checksum}",
                    ),
                    actor_user_id=actor_user_id,
                    scope=scope,
                    batch_number=batch_count,
                    pages=page_batch,
                )
                await self._wait_for_job(submitted, expected_operation="ingest_batch")
                batch_count += 1

            if exported.complete:
                break
            next_cursor = exported.next_cursor
            if not next_cursor or next_cursor in seen_cursors:
                raise SynchronizationError("Novo page export returned an invalid cursor sequence")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        document_checksum = checksum.hexdigest()
        finalized = await self.worker_client.ingest_finalize(
            request_id=new_request_id(),
            idempotency_key=_idempotency_key(
                "ingest-finalize",
                [scope],
                extra=f"{page_count}:{batch_count}:{document_checksum}",
            ),
            actor_user_id=actor_user_id,
            scope=scope,
            document_checksum=document_checksum,
            page_count=page_count,
            batch_count=batch_count,
        )
        await self._wait_for_job(finalized, expected_operation="ingest_finalize")
        return ExportSummary(notebook_id, page_count, batch_count, document_checksum)

    async def _wait_for_job(self, submitted: dict[str, Any], *, expected_operation: str) -> dict[str, Any]:
        job = unwrap_job(submitted)
        job_id = str(job.get("jobId") or "")
        if not job_id or str(job.get("operation") or "") != expected_operation:
            raise WorkerRejected("Worker returned an invalid synchronization job")
        while True:
            state = str(job.get("state") or "").lower()
            if state in {"succeeded", "success", "completed"}:
                return job
            if state in {"failed", "canceled", "cancelled"}:
                error = job.get("error")
                message = error.get("message") if isinstance(error, dict) else None
                raise SynchronizationError(str(message or "Notebook preparation failed"))
            if state not in {"queued", "running"}:
                raise WorkerRejected("Worker returned an invalid synchronization job state")
            await asyncio.sleep(self.settings.worker_poll_interval_s)
            job = unwrap_job(await self.worker_client.job_status(job_id))
            if str(job.get("jobId") or "") != job_id or str(job.get("operation") or "") != expected_operation:
                raise WorkerRejected("Worker returned the wrong synchronization job")


def _page_batch_checksum(pages: list[dict[str, Any]]) -> str:
    # Kept local to avoid trusting an upstream-provided checksum. The
    # WorkerHttpClient independently recomputes this before signing the batch.
    return ingest_pages_checksum(pages)


def _partition_page_batches(
    pages: list[dict[str, Any]],
    *,
    max_canonical_bytes: int,
) -> Iterable[list[dict[str, Any]]]:
    """Partition pages by their exact canonical JSON-array byte length."""

    batch: list[dict[str, Any]] = []
    # The canonical array contributes two brackets plus one comma between each
    # pair of items. Individual page encodings are stable and UTF-8 aware.
    batch_bytes = 2
    for page in pages:
        page_bytes = len(canonical_json(page))
        candidate_bytes = batch_bytes + page_bytes + (1 if batch else 0)
        if candidate_bytes > max_canonical_bytes:
            if not batch:
                raise ExportLimitExceeded("A Novo page exceeds the configured ingest batch byte limit")
            yield batch
            batch = []
            batch_bytes = 2
            candidate_bytes = batch_bytes + page_bytes
            if candidate_bytes > max_canonical_bytes:
                raise ExportLimitExceeded("A Novo page exceeds the configured ingest batch byte limit")
        batch.append(page)
        batch_bytes = candidate_bytes
    if batch:
        yield batch


def _idempotency_key(operation: str, scope: list[dict[str, str]], *, extra: str = "") -> str:
    material = canonical_json({"operation": operation, "scope": scope, "extra": extra})
    return f"sync-{hashlib.sha256(material).hexdigest()}"
