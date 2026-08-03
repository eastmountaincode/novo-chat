from __future__ import annotations

import re
from typing import Any, Protocol

from novo_chat.protocol import JobProgressDetail


_SAFE_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _error_code(value: str, fallback: str) -> str:
    return value if _SAFE_ERROR_CODE.fullmatch(value) else fallback


class WorkerUnavailable(Exception):
    def __init__(
        self,
        message: str = "Authenticated compute transport is unavailable",
        *,
        code: str = "WORKER_UNAVAILABLE",
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = _error_code(code, "WORKER_UNAVAILABLE")
        self.retryable = bool(retryable)


class WorkerRejected(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "WORKER_REJECTED",
        retryable: bool = False,
        status_code: int = 502,
    ) -> None:
        super().__init__(message)
        self.code = _error_code(code, "WORKER_REJECTED")
        self.retryable = bool(retryable)
        self.status_code = status_code if status_code in {400, 404, 409, 413, 422, 429, 502} else 502


class GatewayWorkerClient(Protocol):
    """Narrow interface used by the gateway and its mock-only tests."""

    async def health(self) -> dict[str, Any]: ...

    async def capabilities(self) -> dict[str, Any]: ...

    async def model_status(self) -> dict[str, Any]: ...

    async def index_status(
        self,
        *,
        request_id: str,
        actor_user_id: str,
        scope: list[dict[str, str]],
    ) -> dict[str, Any]: ...

    async def ingest_batch(
        self,
        *,
        request_id: str,
        idempotency_key: str,
        actor_user_id: str,
        scope: dict[str, str],
        batch_number: int,
        pages: list[dict[str, Any]],
    ) -> dict[str, Any]: ...

    async def ingest_finalize(
        self,
        *,
        request_id: str,
        idempotency_key: str,
        actor_user_id: str,
        scope: dict[str, str],
        document_checksum: str,
        page_count: int,
        batch_count: int,
    ) -> dict[str, Any]: ...

    async def submit(
        self,
        *,
        operation: str,
        request_id: str,
        idempotency_key: str,
        actor_user_id: str,
        scope: list[dict[str, str]],
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...

    async def job_status(self, job_id: str) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


def unwrap_job(payload: dict[str, Any]) -> dict[str, Any]:
    job = payload.get("job", payload)
    if not isinstance(job, dict):
        raise WorkerRejected("Worker returned an invalid job response")
    return job


def public_job_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop worker result and internal fields from the polling response."""

    job = unwrap_job(payload)
    error = job.get("error")
    if isinstance(error, dict):
        code = _error_code(str(error.get("code") or ""), "WORKER_JOB_FAILED")
        error = {
            "code": code,
            "message": str(error.get("message") or "The worker job failed.")[:2000],
            "retryable": error.get("retryable") is True,
        }
    elif error:
        error = {
            "code": "WORKER_JOB_FAILED",
            "message": str(error)[:2000],
            "retryable": False,
        }
    progress_detail = None
    if job.get("progressDetail") is not None:
        try:
            progress_detail = JobProgressDetail.model_validate(
                job["progressDetail"]
            ).model_dump(mode="json", by_alias=True)
        except (TypeError, ValueError) as exc:
            raise WorkerRejected("Worker returned invalid progress detail") from exc
    return {
        "jobId": str(job.get("jobId") or job.get("id") or ""),
        "state": str(job.get("state") or "unknown"),
        "operation": str(job.get("operation") or ""),
        "progress": job.get("progress"),
        "error": error or None,
        "createdAt": job.get("createdAt"),
        "updatedAt": job.get("updatedAt"),
        "progressDetail": progress_detail,
    }


def completed_result(payload: dict[str, Any]) -> Any:
    job = unwrap_job(payload)
    state = str(job.get("state") or "").lower()
    if state not in {"completed", "succeeded", "success"}:
        raise WorkerRejected(f"Worker job is not complete (state={state or 'unknown'})")
    if "result" not in job:
        raise WorkerRejected("Completed worker job has no result")
    return job["result"]
