from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from novo_chat.protocol import NotebookScope


DEFAULT_JOB_OWNERSHIP_TTL_S = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class OwnedJob:
    job_id: str
    user_id: str
    operation: str
    notebook_ids: tuple[str, ...]
    scope: tuple[NotebookScope, ...] | None
    created_at: str


def _normalized_scope(
    scope: Sequence[NotebookScope | Mapping[str, Any]],
) -> tuple[NotebookScope, ...]:
    normalized = tuple(
        sorted(
            (
                item if isinstance(item, NotebookScope) else NotebookScope.model_validate(item)
                for item in scope
            ),
            key=lambda item: item.notebook_id,
        )
    )
    notebook_ids = [item.notebook_id for item in normalized]
    if len(notebook_ids) != len(set(notebook_ids)):
        raise ValueError("job ownership scope contains a duplicate notebook")
    return normalized


def _scope_json(scope: Sequence[NotebookScope]) -> str:
    return json.dumps(
        [item.model_dump(mode="json", by_alias=True) for item in scope],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _scope_from_json(value: str | None) -> tuple[NotebookScope, ...] | None:
    # Rows created by the pre-scope schema deliberately remain distinguishable
    # from an empty model-control scope. Their exact revisions cannot be
    # reconstructed safely, so the gateway rejects their release with 409.
    if value is None:
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError("job ownership scope is invalid")
    return _normalized_scope(decoded)


class JobOwnershipStore:
    """Durably bind a Novo identity and exact content scope to a worker job."""

    def __init__(self, path: Path, *, retention_ttl_s: int = DEFAULT_JOB_OWNERSHIP_TTL_S) -> None:
        if retention_ttl_s <= 0:
            raise ValueError("job ownership retention must be positive")
        self.path = path
        self.retention_ttl_s = int(retention_ttl_s)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_owners (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    notebook_ids_json TEXT NOT NULL,
                    scope_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(job_owners)").fetchall()
            }
            if "scope_json" not in columns:
                connection.execute("ALTER TABLE job_owners ADD COLUMN scope_json TEXT")
            connection.execute("CREATE INDEX IF NOT EXISTS job_owners_user_id ON job_owners(user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS job_owners_created_at ON job_owners(created_at)")
            self._purge_expired_connection(connection, datetime.now(timezone.utc))
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    async def add(
        self,
        job_id: str,
        user_id: str,
        operation: str,
        scope: Sequence[NotebookScope | Mapping[str, Any]],
    ) -> OwnedJob:
        created_at = datetime.now(timezone.utc).isoformat()
        normalized_scope = _normalized_scope(scope)
        notebook_ids = tuple(item.notebook_id for item in normalized_scope)
        created_at = await asyncio.to_thread(
            self._add_sync,
            job_id,
            user_id,
            operation,
            notebook_ids,
            normalized_scope,
            created_at,
        )
        return OwnedJob(job_id, user_id, operation, notebook_ids, normalized_scope, created_at)

    def _add_sync(
        self,
        job_id: str,
        user_id: str,
        operation: str,
        notebook_ids: tuple[str, ...],
        scope: tuple[NotebookScope, ...],
        created_at: str,
    ) -> str:
        notebook_ids_json = json.dumps(notebook_ids, separators=(",", ":"))
        scope_json = _scope_json(scope)
        with sqlite3.connect(self.path) as connection:
            self._purge_expired_connection(connection, datetime.now(timezone.utc))
            connection.execute(
                """
                INSERT OR IGNORE INTO job_owners(
                    job_id, user_id, operation, notebook_ids_json, scope_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, user_id, operation, notebook_ids_json, scope_json, created_at),
            )
            existing = connection.execute(
                """
                SELECT user_id, operation, notebook_ids_json, scope_json, created_at
                FROM job_owners WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if not existing or (
                existing[0] != user_id
                or existing[1] != operation
                or tuple(json.loads(existing[2])) != notebook_ids
            ):
                raise RuntimeError("worker job ownership collision")
            existing_scope = _scope_from_json(existing[3])
            if existing_scope is None:
                # A repeated, otherwise identical submission can safely fill in
                # the new revision-aware column after an in-place migration.
                connection.execute(
                    "UPDATE job_owners SET scope_json = ? WHERE job_id = ? AND scope_json IS NULL",
                    (scope_json, job_id),
                )
            elif existing_scope != scope:
                raise RuntimeError("worker job ownership collision")
            return str(existing[4])

    async def get(self, job_id: str) -> OwnedJob | None:
        return await asyncio.to_thread(self._get_sync, job_id)

    def _get_sync(self, job_id: str) -> OwnedJob | None:
        with sqlite3.connect(self.path) as connection:
            self._purge_expired_connection(connection, datetime.now(timezone.utc))
            row = connection.execute(
                """
                SELECT job_id, user_id, operation, notebook_ids_json, scope_json, created_at
                FROM job_owners WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return OwnedJob(
            str(row[0]),
            str(row[1]),
            str(row[2]),
            tuple(json.loads(row[3])),
            _scope_from_json(row[4]),
            str(row[5]),
        )

    async def purge_expired(self) -> int:
        return await asyncio.to_thread(self._purge_expired_sync)

    def _purge_expired_sync(self) -> int:
        with sqlite3.connect(self.path) as connection:
            return self._purge_expired_connection(connection, datetime.now(timezone.utc))

    def _purge_expired_connection(self, connection: sqlite3.Connection, now: datetime) -> int:
        cutoff = (now - timedelta(seconds=self.retention_ttl_s)).isoformat()
        cursor = connection.execute("DELETE FROM job_owners WHERE created_at < ?", (cutoff,))
        return max(0, int(cursor.rowcount))

    async def healthy(self) -> bool:
        try:
            await asyncio.to_thread(self._healthy_sync)
        except (OSError, sqlite3.Error):
            return False
        return True

    def _healthy_sync(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("SELECT 1").fetchone()


__all__ = ["DEFAULT_JOB_OWNERSHIP_TTL_S", "JobOwnershipStore", "OwnedJob"]
