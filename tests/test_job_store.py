from __future__ import annotations

import sqlite3
import tempfile
import unittest

from novo_chat.jobs import IdempotencyConflict, JobNotFound, JobStore
from novo_chat.protocol import JobOperation, JobState, NotebookScope


REQUEST_ID = "123e4567-e89b-42d3-a456-426614174000"


class DurableJobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.store = JobStore(f"{self.temporary_directory.name}/worker.sqlite3")
        self.scope = (
            NotebookScope(
                notebook_id="notebook-a",
                content_revision="sha256:revision-a",
                index_schema_version="index-v1",
            ),
        )

    def submit(self, actor: str, idempotency_key: str, payload=None):
        return self.store.submit(
            environment="staging",
            actor_user_id=actor,
            idempotency_key=idempotency_key,
            request_id=REQUEST_ID,
            operation=JobOperation.INDEX_REBUILD,
            scope=self.scope,
            payload=payload or {"force": False},
            now=1_700_000_000,
        )

    def test_idempotent_retry_and_cross_user_dedupe_have_separate_handles(self):
        first = self.submit("user-a", "idempotency-a")
        retry = self.submit("user-a", "idempotency-a")
        second_user = self.submit("user-b", "idempotency-b")
        self.assertTrue(first.submission_created)
        self.assertFalse(retry.submission_created)
        self.assertEqual(first.job.job_id, retry.job.job_id)
        self.assertNotEqual(first.job.job_id, second_user.job.job_id)
        self.assertEqual(first.job.operation_id, second_user.job.operation_id)
        self.assertFalse(second_user.operation_created)

    def test_reusing_idempotency_key_for_different_request_is_conflict(self):
        self.submit("user-a", "same-key", {"force": False})
        with self.assertRaises(IdempotencyConflict):
            self.submit("user-a", "same-key", {"force": True})

    def test_state_and_nonce_survive_new_store_instance(self):
        outcome = self.submit("user-a", "durable-key")
        self.store.set_running(outcome.job.operation_id, progress=0.25, now=1_700_000_001)
        reopened = JobStore(f"{self.temporary_directory.name}/worker.sqlite3")
        job = reopened.get(outcome.job.job_id, environment="staging")
        self.assertEqual(job.state, JobState.RUNNING)
        self.assertEqual(job.progress, 0.25)
        self.assertTrue(
            reopened.claim(
                environment="staging",
                key_id="gateway",
                nonce="nonce-one",
                seen_at=10,
                expires_at=70,
            )
        )
        self.assertFalse(
            JobStore(f"{self.temporary_directory.name}/worker.sqlite3").claim(
                environment="staging",
                key_id="gateway",
                nonce="nonce-one",
                seen_at=11,
                expires_at=71,
            )
        )

    def test_query_running_during_restart_becomes_retryable_failure(self):
        outcome = self.store.submit(
            environment="staging",
            actor_user_id="user-a",
            idempotency_key="query-key",
            request_id=REQUEST_ID,
            operation=JobOperation.QUERY,
            scope=self.scope,
            payload={"question": "question"},
            now=1_700_000_000,
        )
        self.store.set_running(outcome.job.operation_id, now=1_700_000_001)
        self.store.recover_interrupted_operations(now=1_700_000_002)
        recovered = self.store.get(outcome.job.job_id, environment="staging")
        self.assertEqual(recovered.state, JobState.FAILED)
        self.assertEqual(recovered.error_code, "WORKER_RESTARTED")
        self.assertTrue(recovered.retryable)

    def test_index_catalog_requires_exact_revision_schema_and_environment(self):
        self.store.activate_index(self.scope[0], environment="staging", now=1_700_000_000)
        self.assertEqual(self.store.index_status(self.scope, environment="staging"), [True])
        changed = (
            self.scope[0].model_copy(update={"content_revision": "sha256:revision-b"}),
        )
        self.assertEqual(self.store.index_status(changed, environment="staging"), [False])
        self.assertEqual(self.store.index_status(self.scope, environment="production"), [False])

    def test_terminal_retention_removes_query_payload_result_and_handles_only(self):
        terminal = self.store.submit(
            environment="staging",
            actor_user_id="user-terminal",
            idempotency_key="terminal-query-key",
            request_id=REQUEST_ID,
            operation=JobOperation.QUERY,
            scope=self.scope,
            payload={"question": "sensitive old question", "model": "model:a"},
            now=1_600_000_000,
        )
        self.store.set_running(terminal.job.operation_id, now=1_600_000_001)
        self.store.succeed(
            terminal.job.operation_id,
            {"kind": "query", "answer": "sensitive old answer", "model": "model:a", "citations": []},
            now=1_600_000_002,
        )
        retained_terminal = self.store.submit(
            environment="staging",
            actor_user_id="user-recent",
            idempotency_key="recent-query-key",
            request_id=REQUEST_ID,
            operation=JobOperation.QUERY,
            scope=self.scope,
            payload={"question": "recent question", "model": "model:a"},
            now=1_700_000_000,
        )
        self.store.succeed(
            retained_terminal.job.operation_id,
            {"kind": "query", "answer": "recent answer", "model": "model:a", "citations": []},
            now=1_700_000_001,
        )

        running = self.store.submit(
            environment="staging",
            actor_user_id="user-running",
            idempotency_key="running-query-key",
            request_id=REQUEST_ID,
            operation=JobOperation.QUERY,
            scope=self.scope,
            payload={"question": "active running question", "model": "model:a"},
            now=1_600_000_000,
        )
        self.store.set_running(running.job.operation_id, now=1_600_000_001)
        queued = self.store.submit(
            environment="staging",
            actor_user_id="user-queued",
            idempotency_key="queued-query-key",
            request_id=REQUEST_ID,
            operation=JobOperation.QUERY,
            scope=self.scope,
            payload={"question": "active queued question", "model": "model:a"},
            now=1_600_000_000,
        )
        self.store.activate_index(
            self.scope[0],
            environment="staging",
            artifact_id="artifact-still-active",
            now=1_600_000_000,
        )

        deleted = self.store.purge_terminal_operations(
            environment="staging",
            updated_before=1_650_000_000,
        )

        self.assertEqual(deleted, 1)
        with self.assertRaises(JobNotFound):
            self.store.get(terminal.job.job_id, environment="staging")
        self.assertEqual(
            self.store.get(retained_terminal.job.job_id, environment="staging").state,
            JobState.SUCCEEDED,
        )
        self.assertEqual(self.store.get(running.job.job_id, environment="staging").state, JobState.RUNNING)
        self.assertEqual(self.store.get(queued.job.job_id, environment="staging").state, JobState.QUEUED)
        active_index = self.store.active_index(self.scope[0], environment="staging")
        self.assertIsNotNone(active_index)
        self.assertEqual(active_index.artifact_id, "artifact-still-active")

        with sqlite3.connect(self.store.database_path) as connection:
            operation = connection.execute(
                "SELECT payload_json, result_json FROM operations WHERE operation_id = ?",
                (terminal.job.operation_id,),
            ).fetchone()
            submission = connection.execute(
                "SELECT 1 FROM submissions WHERE submission_id = ?",
                (terminal.job.job_id,),
            ).fetchone()
        self.assertIsNone(operation)
        self.assertIsNone(submission)


if __name__ == "__main__":
    unittest.main()
