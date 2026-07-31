from __future__ import annotations

from typing import Any, TypeVar
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from novo_chat.protocol import (
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    IndexStatusRequest,
    IndexStatusResponse,
    IngestBatchRequest,
    IngestFinalizeRequest,
    JobStatusResponse,
    JobSubmissionResponse,
    ModelJobRequest,
    ModelStatusResponse,
    NotebookScope,
    PageDocument,
    ProtocolError,
    QueryRequest,
    RebuildRequest,
    canonical_json,
    ingest_pages_checksum,
    new_request_id,
    sign_request,
    verify_response,
)

from .config import GatewaySettings
from .secrets import FileSecret
from .worker_client import WorkerRejected, WorkerUnavailable


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class WorkerHttpClient:
    """Authenticated client for the loopback reverse-tunnel listener.

    It accepts typed job data only. In particular, it has no cookie jar and no
    parameter capable of receiving a browser request or Novo session.
    """

    def __init__(
        self,
        *,
        base_url: str,
        request_secret: FileSecret,
        response_secret: FileSecret,
        request_key_id: str,
        response_key_id: str,
        environment: str,
        timeout_s: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        self.base_url = base_url.rstrip("/")
        self.base_path = parsed.path.rstrip("/")
        self.request_secret = request_secret
        self.response_secret = response_secret
        self.request_key_id = request_key_id
        self.response_key_id = response_key_id
        self.environment = environment
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, trust_env=False, follow_redirects=False)

    @classmethod
    def from_settings(cls, settings: GatewaySettings) -> "WorkerHttpClient":
        return cls(
            base_url=settings.worker_base_url,
            request_secret=FileSecret(settings.worker_signing_secret_file, minimum_bytes=32),
            response_secret=FileSecret(settings.effective_worker_response_secret_file, minimum_bytes=32),
            request_key_id=settings.worker_request_key_id,
            response_key_id=settings.worker_response_key_id,
            environment=settings.environment,
            timeout_s=settings.request_timeout_s,
        )

    async def health(self) -> dict[str, Any]:
        response = await self._request("GET", "/health", HealthResponse)
        payload = response.model_dump(mode="json", by_alias=True)
        payload["ok"] = response.state.value == "ready" and response.queue_healthy and response.index_service_healthy
        return payload

    async def capabilities(self) -> dict[str, Any]:
        response = await self._request("GET", "/capabilities", CapabilitiesResponse)
        payload = response.model_dump(mode="json", by_alias=True)
        # The public UI consumes a small generic model list while the complete
        # signed capabilities object remains available to the gateway.
        payload["models"] = list(response.approved_models)
        return payload

    async def model_status(self) -> dict[str, Any]:
        response = await self._request("GET", "/models/status", ModelStatusResponse)
        return {
            "models": {
                row.model: {"state": row.state.value, "healthy": row.state.value == "ready"}
                for row in response.models
            }
        }

    async def index_status(
        self,
        *,
        request_id: str,
        actor_user_id: str,
        scope: list[dict[str, str]],
    ) -> dict[str, Any]:
        try:
            model = IndexStatusRequest(
                request_id=request_id,
                actor_user_id=actor_user_id,
                scope=tuple(NotebookScope.model_validate(row) for row in scope),
            )
        except (ValidationError, ValueError) as exc:
            raise WorkerRejected("Index status did not satisfy the compute protocol") from exc
        response = await self._request("POST", "/indexes/status", IndexStatusResponse, request_model=model)
        return response.model_dump(mode="json", by_alias=True)

    async def ingest_batch(
        self,
        *,
        request_id: str,
        idempotency_key: str,
        actor_user_id: str,
        scope: dict[str, str],
        batch_number: int,
        pages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            page_models = tuple(PageDocument.model_validate(page) for page in pages)
            model = IngestBatchRequest(
                request_id=request_id,
                idempotency_key=idempotency_key,
                actor_user_id=actor_user_id,
                scope=(NotebookScope.model_validate(scope),),
                batch_number=batch_number,
                batch_checksum=ingest_pages_checksum(page_models),
                pages=page_models,
            )
        except (ValidationError, ValueError) as exc:
            raise WorkerRejected("Ingest batch did not satisfy the compute protocol") from exc
        response = await self._request("POST", "/ingest/batches", JobSubmissionResponse, request_model=model)
        return response.model_dump(mode="json", by_alias=True)

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
    ) -> dict[str, Any]:
        try:
            model = IngestFinalizeRequest(
                request_id=request_id,
                idempotency_key=idempotency_key,
                actor_user_id=actor_user_id,
                scope=(NotebookScope.model_validate(scope),),
                document_checksum=document_checksum,
                page_count=page_count,
                batch_count=batch_count,
            )
        except (ValidationError, ValueError) as exc:
            raise WorkerRejected("Ingest finalization did not satisfy the compute protocol") from exc
        response = await self._request("POST", "/ingest/finalize", JobSubmissionResponse, request_model=model)
        return response.model_dump(mode="json", by_alias=True)

    async def submit(
        self,
        *,
        operation: str,
        request_id: str,
        idempotency_key: str,
        actor_user_id: str,
        scope: list[dict[str, str]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            common = {
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "actor_user_id": actor_user_id,
                "scope": tuple(NotebookScope.model_validate(row) for row in scope),
            }
            if operation == "query":
                strategy = str(payload.get("strategy") or "hybrid")
                if strategy == "auto":
                    strategy = "hybrid"
                elif strategy == "fts":
                    strategy = "lexical"
                model = QueryRequest(
                    **common,
                    question=payload.get("question"),
                    retrieval_question=payload.get("retrieval_question"),
                    model=payload.get("model"),
                    strategy=strategy,
                    max_sources=payload.get("max_sources", 6),
                )
                route = "/query"
            elif operation == "index_rebuild":
                model = RebuildRequest(**common, force=bool(payload.get("force", False)))
                route = "/indexes/rebuild"
            elif operation in {"model_start", "model_stop"}:
                model = ModelJobRequest(**common, model=payload.get("model"))
                route = "/models/start" if operation == "model_start" else "/models/stop"
            else:
                raise WorkerRejected(f"Unsupported worker operation: {operation}")
        except (ValidationError, ValueError) as exc:
            raise WorkerRejected("Job did not satisfy the compute protocol") from exc
        response = await self._request("POST", route, JobSubmissionResponse, request_model=model)
        return response.model_dump(mode="json", by_alias=True)

    async def job_status(self, job_id: str) -> dict[str, Any]:
        if not job_id or len(job_id) > 192:
            raise WorkerRejected("Invalid worker job identifier")
        response = await self._request(
            "GET",
            f"/jobs/{quote(job_id, safe='-._~')}",
            JobStatusResponse,
        )
        return response.model_dump(mode="json", by_alias=True)

    async def _request(
        self,
        method: str,
        route: str,
        response_model: type[ResponseModel],
        *,
        request_model: BaseModel | None = None,
    ) -> ResponseModel:
        request_id = request_model.request_id if request_model is not None else new_request_id()
        body = canonical_json(request_model) if request_model is not None else b""
        signed_path = f"{self.base_path}{route}"
        try:
            headers = sign_request(
                secret=self.request_secret.read_bytes(),
                key_id=self.request_key_id,
                environment=self.environment,
                request_id=request_id,
                method=method,
                path=signed_path,
                body=body,
            )
            if body:
                headers["Content-Type"] = "application/json"
            response = await self._client.request(method, f"{self.base_url}{route}", headers=headers, content=body)
            response_body = response.content
            verify_response(
                headers=response.headers,
                secrets_by_key_id={self.response_key_id: self.response_secret.read_bytes()},
                expected_environment=self.environment,
                request_id=request_id,
                status_code=response.status_code,
                body=response_body,
            )
        except (httpx.HTTPError, OSError, RuntimeError, ProtocolError, ValueError) as exc:
            raise WorkerUnavailable("Authenticated compute transport is unavailable") from exc

        if response.status_code >= 400:
            code = "WORKER_REJECTED"
            retryable = False
            try:
                error = ErrorResponse.model_validate_json(response_body).error
                message = error.message
                code = error.code
                retryable = error.retryable
            except (ValidationError, ValueError):
                message = "Compute worker rejected the request"
            if response.status_code >= 500:
                raise WorkerUnavailable(message, code=code, retryable=retryable)
            public_status = response.status_code if response.status_code in {400, 404, 409, 413, 422, 429} else 502
            raise WorkerRejected(
                message,
                code=code,
                retryable=retryable,
                status_code=public_status,
            )
        try:
            return response_model.model_validate_json(response_body)
        except (ValidationError, ValueError) as exc:
            raise WorkerRejected("Worker returned an invalid signed response") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
