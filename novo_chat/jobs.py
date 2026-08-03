"""Durable worker operations, caller-specific submission handles, and index catalog."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import JobOperation, JobState, NotebookScope, canonical_json


class JobStoreError(Exception):
    pass


class JobNotFound(JobStoreError):
    pass


class IdempotencyConflict(JobStoreError):
    pass


class InvalidJobTransition(JobStoreError):
    pass


class QueueFull(JobStoreError):
    pass


@dataclass(frozen=True)
class StoredJob:
    job_id: str
    operation_id: str
    environment: str
    actor_user_id: str
    request_id: str
    operation: JobOperation
    state: JobState
    created_at: str
    updated_at: str
    progress: float
    progress_detail: dict[str, Any] | None
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    retryable: bool


@dataclass(frozen=True)
class StoredOperation:
    operation_id: str
    environment: str
    actor_user_id: str
    request_id: str
    operation: JobOperation
    scope: tuple[NotebookScope, ...]
    payload: dict[str, Any]
    state: JobState
    progress: float
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ActiveIndex:
    scope: NotebookScope
    artifact_id: str
    activated_at: str


@dataclass(frozen=True)
class SubmissionOutcome:
    job: StoredJob
    submission_created: bool
    operation_created: bool


@dataclass(frozen=True)
class StorageRetentionSnapshot:
    """Filesystem references captured atomically with active-index expiry."""

    idle: bool
    expired_indexes: int
    active_scopes: tuple[NotebookScope, ...]
    active_artifact_ids: tuple[str, ...]


def _utc_timestamp(epoch_seconds: float | int) -> str:
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _new_handle(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(24)}"


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, JobOperation):
        return value.value
    return value


def _scope_json(scope: Sequence[NotebookScope]) -> list[dict[str, Any]]:
    return [entry.model_dump(mode="json", by_alias=True) for entry in scope]


class JobStore:
    """SQLite state for durable operations and opaque per-caller handles.

    An operation may be shared by multiple submissions, but each submission has
    its own unpredictable public job ID and actor binding.  Internal operation
    IDs never need to cross the worker API boundary.
    """

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    environment TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0.0 CHECK(progress >= 0.0 AND progress <= 1.0),
                    progress_detail_json TEXT,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    retryable INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(environment, operation_type, dedupe_key)
                );

                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    environment TEXT NOT NULL,
                    actor_user_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    operation_id TEXT NOT NULL REFERENCES operations(operation_id),
                    created_at TEXT NOT NULL,
                    UNIQUE(environment, actor_user_id, idempotency_key)
                );

                CREATE INDEX IF NOT EXISTS submissions_operation_idx
                    ON submissions(operation_id);

                CREATE TABLE IF NOT EXISTS replay_nonces (
                    environment TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY(environment, key_id, nonce)
                );

                CREATE INDEX IF NOT EXISTS replay_nonces_expiry_idx
                    ON replay_nonces(expires_at);

                CREATE TABLE IF NOT EXISTS active_indexes (
                    environment TEXT NOT NULL,
                    notebook_id TEXT NOT NULL,
                    index_schema_version TEXT NOT NULL,
                    content_revision TEXT NOT NULL,
                    artifact_id TEXT NOT NULL DEFAULT '',
                    activated_at TEXT NOT NULL,
                    PRIMARY KEY(environment, notebook_id, index_schema_version)
                );
                """
            )
            active_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(active_indexes)").fetchall()
            }
            if "artifact_id" not in active_columns:
                connection.execute(
                    "ALTER TABLE active_indexes ADD COLUMN artifact_id TEXT NOT NULL DEFAULT ''"
                )
            operation_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(operations)").fetchall()
            }
            if "progress_detail_json" not in operation_columns:
                connection.execute("ALTER TABLE operations ADD COLUMN progress_detail_json TEXT")
            # The table-level unique constraint is retained for compatibility,
            # but terminal operations vacate their live dedupe key. This makes
            # deduplication active-only while preserving old submission handles.
            connection.execute(
                """
                UPDATE operations
                SET dedupe_key = dedupe_key || ':terminal:' || operation_id
                WHERE state IN (?, ?, ?)
                  AND instr(dedupe_key, ':terminal:' || operation_id) = 0
                """,
                (JobState.SUCCEEDED.value, JobState.FAILED.value, JobState.CANCELED.value),
            )

    def ping(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False

    def claim(self, *, environment: str, key_id: str, nonce: str, seen_at: int, expires_at: int) -> bool:
        """Durably claim a signed-request nonce for replay prevention."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM replay_nonces WHERE expires_at < ?", (int(seen_at),))
            try:
                connection.execute(
                    "INSERT INTO replay_nonces(environment, key_id, nonce, expires_at) VALUES (?, ?, ?, ?)",
                    (environment, key_id, nonce, int(expires_at)),
                )
            except sqlite3.IntegrityError:
                connection.rollback()
                return False
            connection.commit()
            return True

    def submit(
        self,
        *,
        environment: str,
        actor_user_id: str,
        idempotency_key: str,
        request_id: str,
        operation: JobOperation,
        scope: Sequence[NotebookScope],
        payload: Mapping[str, Any],
        dedupe_key: str | None = None,
        max_active_operations: int | None = None,
        now: float | int | None = None,
    ) -> SubmissionOutcome:
        timestamp = time.time() if now is None else float(now)
        timestamp_text = _utc_timestamp(timestamp)
        scope_value = _scope_json(scope)
        payload_value = {str(key): _json_value(value) for key, value in payload.items()}
        request_document = {
            "operation": operation.value,
            "scope": scope_value,
            "payload": payload_value,
        }
        request_fingerprint = hashlib.sha256(canonical_json(request_document)).hexdigest()
        actual_dedupe_key = dedupe_key or request_fingerprint

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_submission = connection.execute(
                """
                SELECT submission_id, request_fingerprint
                FROM submissions
                WHERE environment = ? AND actor_user_id = ? AND idempotency_key = ?
                """,
                (environment, actor_user_id, idempotency_key),
            ).fetchone()
            if existing_submission is not None:
                if existing_submission["request_fingerprint"] != request_fingerprint:
                    connection.rollback()
                    raise IdempotencyConflict("idempotency key was already used for a different request")
                job = self._get_with_connection(
                    connection,
                    str(existing_submission["submission_id"]),
                    environment=environment,
                )
                connection.commit()
                return SubmissionOutcome(job=job, submission_created=False, operation_created=False)

            existing_operation = connection.execute(
                """
                SELECT operation_id
                FROM operations
                WHERE environment = ? AND operation_type = ? AND dedupe_key = ?
                """,
                (environment, operation.value, actual_dedupe_key),
            ).fetchone()
            operation_created = existing_operation is None
            if operation_created:
                if max_active_operations is not None:
                    active_count = int(
                        connection.execute(
                            """
                            SELECT COUNT(*)
                            FROM operations
                            WHERE environment = ? AND state IN (?, ?)
                            """,
                            (environment, JobState.QUEUED.value, JobState.RUNNING.value),
                        ).fetchone()[0]
                    )
                    if active_count >= max_active_operations:
                        connection.rollback()
                        raise QueueFull("worker queue is full")
                operation_id = _new_handle("op")
                connection.execute(
                    """
                    INSERT INTO operations(
                        operation_id, environment, operation_type, dedupe_key,
                        scope_json, payload_json, state, progress, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, ?, ?)
                    """,
                    (
                        operation_id,
                        environment,
                        operation.value,
                        actual_dedupe_key,
                        canonical_json(scope_value).decode("utf-8"),
                        canonical_json(payload_value).decode("utf-8"),
                        JobState.QUEUED.value,
                        timestamp_text,
                        timestamp_text,
                    ),
                )
            else:
                operation_id = str(existing_operation["operation_id"])

            submission_id = _new_handle("job")
            connection.execute(
                """
                INSERT INTO submissions(
                    submission_id, environment, actor_user_id, idempotency_key,
                    request_id, request_fingerprint, operation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    submission_id,
                    environment,
                    actor_user_id,
                    idempotency_key,
                    request_id,
                    request_fingerprint,
                    operation_id,
                    timestamp_text,
                ),
            )
            job = self._get_with_connection(connection, submission_id, environment=environment)
            connection.commit()
            return SubmissionOutcome(job=job, submission_created=True, operation_created=operation_created)

    def _get_with_connection(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        *,
        environment: str,
    ) -> StoredJob:
        row = connection.execute(
            """
            SELECT
                s.submission_id, s.environment, s.actor_user_id, s.request_id,
                s.operation_id, s.created_at AS submission_created_at,
                o.operation_type, o.state, o.progress, o.progress_detail_json,
                o.result_json,
                o.error_code, o.error_message, o.retryable,
                o.created_at AS operation_created_at, o.updated_at
            FROM submissions AS s
            JOIN operations AS o ON o.operation_id = s.operation_id
            WHERE s.submission_id = ? AND s.environment = ?
            """,
            (job_id, environment),
        ).fetchone()
        if row is None:
            raise JobNotFound("job not found")
        result = json.loads(row["result_json"]) if row["result_json"] else None
        progress_detail = (
            json.loads(row["progress_detail_json"]) if row["progress_detail_json"] else None
        )
        return StoredJob(
            job_id=str(row["submission_id"]),
            operation_id=str(row["operation_id"]),
            environment=str(row["environment"]),
            actor_user_id=str(row["actor_user_id"]),
            request_id=str(row["request_id"]),
            operation=JobOperation(str(row["operation_type"])),
            state=JobState(str(row["state"])),
            created_at=str(row["submission_created_at"]),
            updated_at=str(row["updated_at"]),
            progress=float(row["progress"]),
            progress_detail=progress_detail,
            result=result,
            error_code=str(row["error_code"]) if row["error_code"] else None,
            error_message=str(row["error_message"]) if row["error_message"] else None,
            retryable=bool(row["retryable"]),
        )

    def get(self, job_id: str, *, environment: str) -> StoredJob:
        with self._connect() as connection:
            return self._get_with_connection(connection, job_id, environment=environment)

    def claim_next_operation(
        self,
        *,
        environment: str,
        now: float | int | None = None,
    ) -> StoredOperation | None:
        """Atomically claim the oldest queued operation for one serialized runner."""

        timestamp = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            running = connection.execute(
                "SELECT 1 FROM operations WHERE environment = ? AND state = ? LIMIT 1",
                (environment, JobState.RUNNING.value),
            ).fetchone()
            if running is not None:
                connection.commit()
                return None
            row = connection.execute(
                """
                SELECT rowid, *
                FROM operations
                WHERE environment = ? AND state = ?
                ORDER BY rowid
                LIMIT 1
                """,
                (environment, JobState.QUEUED.value),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            operation_id = str(row["operation_id"])
            submission = connection.execute(
                """
                SELECT actor_user_id, request_id
                FROM submissions
                WHERE operation_id = ?
                ORDER BY rowid
                LIMIT 1
                """,
                (operation_id,),
            ).fetchone()
            if submission is None:
                connection.rollback()
                raise JobStoreError("queued operation has no submission")
            updated = connection.execute(
                """
                UPDATE operations
                SET state = ?, progress = 0.0, progress_detail_json = NULL,
                    updated_at = ?
                WHERE operation_id = ? AND state = ?
                """,
                (
                    JobState.RUNNING.value,
                    _utc_timestamp(timestamp),
                    operation_id,
                    JobState.QUEUED.value,
                ),
            ).rowcount
            if updated != 1:
                connection.rollback()
                return None
            connection.commit()
            return StoredOperation(
                operation_id=operation_id,
                environment=str(row["environment"]),
                actor_user_id=str(submission["actor_user_id"]),
                request_id=str(submission["request_id"]),
                operation=JobOperation(str(row["operation_type"])),
                scope=tuple(NotebookScope.model_validate(item) for item in json.loads(row["scope_json"])),
                payload=dict(json.loads(row["payload_json"])),
                state=JobState.RUNNING,
                progress=0.0,
                created_at=str(row["created_at"]),
                updated_at=_utc_timestamp(timestamp),
            )

    def get_operation(self, operation_id: str) -> StoredOperation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if row is None:
                raise JobNotFound("operation not found")
            submission = connection.execute(
                """
                SELECT actor_user_id, request_id
                FROM submissions
                WHERE operation_id = ?
                ORDER BY rowid
                LIMIT 1
                """,
                (operation_id,),
            ).fetchone()
            if submission is None:
                raise JobNotFound("operation submission not found")
            return StoredOperation(
                operation_id=operation_id,
                environment=str(row["environment"]),
                actor_user_id=str(submission["actor_user_id"]),
                request_id=str(submission["request_id"]),
                operation=JobOperation(str(row["operation_type"])),
                scope=tuple(NotebookScope.model_validate(item) for item in json.loads(row["scope_json"])),
                payload=dict(json.loads(row["payload_json"])),
                state=JobState(str(row["state"])),
                progress=float(row["progress"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )

    def set_running(self, operation_id: str, *, progress: float = 0.0, now: float | int | None = None) -> None:
        self._transition(
            operation_id,
            allowed_from=(JobState.QUEUED,),
            state=JobState.RUNNING,
            progress=progress,
            now=now,
        )

    def set_progress(
        self,
        operation_id: str,
        progress: float,
        *,
        progress_detail: Mapping[str, Any] | None = None,
        now: float | int | None = None,
    ) -> None:
        self._transition(
            operation_id,
            allowed_from=(JobState.RUNNING,),
            state=JobState.RUNNING,
            progress=progress,
            progress_detail=progress_detail,
            now=now,
        )

    def succeed(self, operation_id: str, result: Mapping[str, Any], *, now: float | int | None = None) -> None:
        self._transition(
            operation_id,
            allowed_from=(JobState.QUEUED, JobState.RUNNING),
            state=JobState.SUCCEEDED,
            progress=1.0,
            result=result,
            now=now,
        )

    def fail(
        self,
        operation_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
        now: float | int | None = None,
    ) -> None:
        self._transition(
            operation_id,
            allowed_from=(JobState.QUEUED, JobState.RUNNING),
            state=JobState.FAILED,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            now=now,
        )

    def _transition(
        self,
        operation_id: str,
        *,
        allowed_from: Iterable[JobState],
        state: JobState,
        progress: float | None = None,
        progress_detail: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        retryable: bool = False,
        now: float | int | None = None,
    ) -> None:
        if progress is not None and not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0 and 1")
        timestamp = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise JobNotFound("operation not found")
            allowed_values = {item.value for item in allowed_from}
            if str(row["state"]) not in allowed_values:
                connection.rollback()
                raise InvalidJobTransition("job state transition is not permitted")
            connection.execute(
                """
                UPDATE operations
                SET state = ?, progress = COALESCE(?, progress),
                    progress_detail_json = COALESCE(?, progress_detail_json),
                    result_json = ?, error_code = ?, error_message = ?,
                    retryable = ?, updated_at = ?
                    , dedupe_key = CASE
                        WHEN ? IN (?, ?, ?)
                        THEN dedupe_key || ':terminal:' || operation_id
                        ELSE dedupe_key
                      END
                WHERE operation_id = ?
                """,
                (
                    state.value,
                    progress,
                    (
                        canonical_json(dict(progress_detail)).decode("utf-8")
                        if progress_detail is not None else None
                    ),
                    canonical_json(dict(result)).decode("utf-8") if result is not None else None,
                    error_code,
                    error_message,
                    int(retryable),
                    _utc_timestamp(timestamp),
                    state.value,
                    JobState.SUCCEEDED.value,
                    JobState.FAILED.value,
                    JobState.CANCELED.value,
                    operation_id,
                ),
            )
            connection.commit()

    def recover_interrupted_operations(self, *, now: float | int | None = None) -> None:
        """Recover only operations whose idempotency permits safe resumption."""

        timestamp = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE operations
                SET state = ?, progress = 0.0, updated_at = ?
                WHERE state = ? AND operation_type IN (?, ?, ?)
                """,
                (
                    JobState.QUEUED.value,
                    _utc_timestamp(timestamp),
                    JobState.RUNNING.value,
                    JobOperation.INGEST_BATCH.value,
                    JobOperation.INGEST_FINALIZE.value,
                    JobOperation.INDEX_REBUILD.value,
                ),
            )
            connection.execute(
                """
                UPDATE operations
                SET state = ?, error_code = ?, error_message = ?, retryable = 1,
                    updated_at = ?, dedupe_key = dedupe_key || ':terminal:' || operation_id
                WHERE state = ? AND operation_type IN (?, ?, ?)
                """,
                (
                    JobState.FAILED.value,
                    "WORKER_RESTARTED",
                    "The worker restarted before the operation completed.",
                    _utc_timestamp(timestamp),
                    JobState.RUNNING.value,
                    JobOperation.QUERY.value,
                    JobOperation.MODEL_START.value,
                    JobOperation.MODEL_STOP.value,
                ),
            )
            connection.commit()

    def purge_terminal_operations(
        self,
        *,
        environment: str,
        updated_before: float | int,
    ) -> int:
        """Atomically remove expired terminal operations and all caller handles.

        Active operations and the independent active-index catalog are never
        considered by this deletion. The terminal transition timestamp, rather
        than submission time, starts the retention window.
        """

        cutoff = _utc_timestamp(float(updated_before))
        terminal_states = (
            JobState.SUCCEEDED.value,
            JobState.FAILED.value,
            JobState.CANCELED.value,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                DELETE FROM submissions
                WHERE operation_id IN (
                    SELECT operation_id
                    FROM operations
                    WHERE environment = ?
                      AND state IN (?, ?, ?)
                      AND julianday(updated_at) < julianday(?)
                )
                """,
                (environment, *terminal_states, cutoff),
            )
            deleted = connection.execute(
                """
                DELETE FROM operations
                WHERE environment = ?
                  AND state IN (?, ?, ?)
                  AND julianday(updated_at) < julianday(?)
                """,
                (environment, *terminal_states, cutoff),
            ).rowcount
            connection.commit()
        return int(deleted)

    def active_operation_count(self, *, environment: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM operations
                    WHERE environment = ? AND state IN (?, ?)
                    """,
                    (environment, JobState.QUEUED.value, JobState.RUNNING.value),
                ).fetchone()[0]
            )

    def storage_retention_snapshot(
        self,
        *,
        environment: str,
        activated_before: float | int,
    ) -> StorageRetentionSnapshot:
        """Expire old index pointers and capture the remaining storage keep set.

        Expiry is performed only while the environment has no queued or
        running operation. The idle check, deletion, and reference snapshot
        share one immediate SQLite transaction so filesystem maintenance never
        races an already-recorded operation.
        """

        cutoff = _utc_timestamp(float(activated_before))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active_operations = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM operations
                    WHERE environment = ? AND state IN (?, ?)
                    """,
                    (environment, JobState.QUEUED.value, JobState.RUNNING.value),
                ).fetchone()[0]
            )
            expired = 0
            if active_operations == 0:
                expired = int(
                    connection.execute(
                        """
                        DELETE FROM active_indexes
                        WHERE environment = ?
                          AND julianday(activated_at) < julianday(?)
                        """,
                        (environment, cutoff),
                    ).rowcount
                )
            rows = connection.execute(
                """
                SELECT notebook_id, content_revision, index_schema_version, artifact_id
                FROM active_indexes
                WHERE environment = ?
                ORDER BY notebook_id, index_schema_version
                """,
                (environment,),
            ).fetchall()
            connection.commit()
        scopes = tuple(
            NotebookScope(
                notebook_id=str(row["notebook_id"]),
                content_revision=str(row["content_revision"]),
                index_schema_version=str(row["index_schema_version"]),
            )
            for row in rows
        )
        return StorageRetentionSnapshot(
            idle=active_operations == 0,
            expired_indexes=expired,
            active_scopes=scopes,
            active_artifact_ids=tuple(str(row["artifact_id"]) for row in rows if row["artifact_id"]),
        )

    def activate_index(
        self,
        scope: NotebookScope,
        *,
        environment: str,
        artifact_id: str = "",
        now: float | int | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO active_indexes(
                    environment, notebook_id, index_schema_version, content_revision,
                    artifact_id, activated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(environment, notebook_id, index_schema_version)
                DO UPDATE SET content_revision = excluded.content_revision,
                              artifact_id = excluded.artifact_id,
                              activated_at = excluded.activated_at
                """,
                (
                    environment,
                    scope.notebook_id,
                    scope.index_schema_version,
                    scope.content_revision,
                    artifact_id,
                    _utc_timestamp(timestamp),
                ),
            )

    def activate_indexes(
        self,
        entries: Sequence[tuple[NotebookScope, str]],
        *,
        environment: str,
        now: float | int | None = None,
    ) -> None:
        """Atomically change every active index pointer after artifacts validate."""

        timestamp = time.time() if now is None else float(now)
        timestamp_text = _utc_timestamp(timestamp)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for scope, artifact_id in entries:
                connection.execute(
                    """
                    INSERT INTO active_indexes(
                        environment, notebook_id, index_schema_version, content_revision,
                        artifact_id, activated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(environment, notebook_id, index_schema_version)
                    DO UPDATE SET content_revision = excluded.content_revision,
                                  artifact_id = excluded.artifact_id,
                                  activated_at = excluded.activated_at
                    """,
                    (
                        environment,
                        scope.notebook_id,
                        scope.index_schema_version,
                        scope.content_revision,
                        artifact_id,
                        timestamp_text,
                    ),
                )
            connection.commit()

    def active_index(self, scope: NotebookScope, *, environment: str) -> ActiveIndex | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT content_revision, artifact_id, activated_at
                FROM active_indexes
                WHERE environment = ? AND notebook_id = ? AND index_schema_version = ?
                """,
                (environment, scope.notebook_id, scope.index_schema_version),
            ).fetchone()
        if row is None or str(row["content_revision"]) != scope.content_revision:
            return None
        return ActiveIndex(
            scope=scope,
            artifact_id=str(row["artifact_id"]),
            activated_at=str(row["activated_at"]),
        )

    def index_status(self, scope: Sequence[NotebookScope], *, environment: str) -> list[bool]:
        status: list[bool] = []
        with self._connect() as connection:
            for entry in scope:
                row = connection.execute(
                    """
                    SELECT content_revision
                    FROM active_indexes
                    WHERE environment = ? AND notebook_id = ? AND index_schema_version = ?
                    """,
                    (environment, entry.notebook_id, entry.index_schema_version),
                ).fetchone()
                status.append(row is not None and str(row["content_revision"]) == entry.content_revision)
        return status


__all__ = [
    "IdempotencyConflict",
    "ActiveIndex",
    "InvalidJobTransition",
    "JobNotFound",
    "JobStore",
    "JobStoreError",
    "QueueFull",
    "StoredJob",
    "StoredOperation",
    "StorageRetentionSnapshot",
    "SubmissionOutcome",
]
