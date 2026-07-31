from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from novo_chat.gateway.store import JobOwnershipStore


SCOPE = [
    {
        "notebookId": "notebook-a",
        "contentRevision": "sha256:revision-a",
        "indexSchemaVersion": "novo-chat-index-v1",
    }
]


def create_legacy_store(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE job_owners (
                job_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                notebook_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO job_owners VALUES (?, ?, ?, ?, ?)",
            (
                "legacy-job",
                "alice",
                "query",
                json.dumps(["notebook-a"]),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def test_store_migrates_legacy_schema_without_inventing_revisions(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    create_legacy_store(path)
    store = JobOwnershipStore(path)
    asyncio.run(store.initialize())

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(job_owners)")}
    assert "scope_json" in columns
    legacy = asyncio.run(store.get("legacy-job"))
    assert legacy is not None
    assert legacy.notebook_ids == ("notebook-a",)
    assert legacy.scope is None

    upgraded = asyncio.run(store.add("legacy-job", "alice", "query", SCOPE))
    assert upgraded.scope is not None
    reloaded = asyncio.run(store.get("legacy-job"))
    assert reloaded is not None and reloaded.scope is not None
    assert reloaded.scope[0].content_revision == "sha256:revision-a"
    assert reloaded.scope[0].index_schema_version == "novo-chat-index-v1"


def test_store_opportunistically_purges_expired_job_mappings(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = JobOwnershipStore(path, retention_ttl_s=60)
    asyncio.run(store.initialize())
    asyncio.run(store.add("expired-job", "alice", "query", SCOPE))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE job_owners SET created_at = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", "expired-job"),
        )

    assert asyncio.run(store.get("expired-job")) is None
