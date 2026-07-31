from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from novo_chat.protocol import validate_same_origin_path

from .config import GatewaySettings
from .models import GatewayJobRequest, GatewayOperation, JobAccepted, JobPublicStatus, WorkerAvailability
from .novo_client import (
    NovoContext,
    NovoIntegrationClient,
    NovoIntegrationUnavailable,
    NovoNotebookUnavailable,
    NovoRevisionChanged,
    NovoUnauthenticated,
)
from .secrets import FileSecret
from .security import csrf_token, login_url, require_same_origin_json_csrf
from .store import JobOwnershipStore, OwnedJob
from .sync import ExportLimitExceeded, GatewaySynchronizer, SynchronizationError, SynchronizationTimeout
from .worker_client import (
    GatewayWorkerClient,
    WorkerRejected,
    WorkerUnavailable,
    completed_result,
    public_job_status,
    unwrap_job,
)


WEB_DIR = Path(__file__).with_name("web")
INDEX_HTML = WEB_DIR / "index.html"
IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,191}$")


async def read_bounded_json_body(request: Request, maximum_bytes: int) -> bytes:
    """Read one strictly framed JSON body without exceeding the configured cap."""

    if request.headers.getlist("transfer-encoding"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transfer-Encoding is not supported",
        )
    content_lengths = request.headers.getlist("content-length")
    if len(content_lengths) != 1 or "," in content_lengths[0]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A single Content-Length header is required",
        )
    raw_length = content_lengths[0].strip()
    if not re.fullmatch(r"[0-9]+", raw_length):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Content-Length")
    normalized_length = raw_length.lstrip("0") or "0"
    maximum = str(maximum_bytes)
    if len(normalized_length) > len(maximum) or (
        len(normalized_length) == len(maximum) and normalized_length > maximum
    ):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request body exceeds the configured limit",
        )
    declared_length = int(normalized_length)

    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > maximum_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request body exceeds the configured limit",
                )
            chunks.append(chunk)
    except ClientDisconnect as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body is incomplete") from exc
    if total != declared_length:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content-Length does not match body")
    body = b"".join(chunks)
    request._body = body
    return body


def create_app(
    settings: GatewaySettings | None = None,
    *,
    novo_client: NovoIntegrationClient | Any | None = None,
    worker_client: GatewayWorkerClient | Any | None = None,
    job_store: JobOwnershipStore | None = None,
) -> FastAPI:
    settings = settings or GatewaySettings.from_env()
    novo_secret = FileSecret(settings.novo_integration_secret_file)
    csrf_secret = FileSecret(settings.effective_csrf_secret_file)
    owned_novo_client = novo_client is None
    if novo_client is None:
        novo_client = NovoIntegrationClient(
            base_url=settings.novo_api_base_url,
            secret=novo_secret,
            session_cookie_name=settings.novo_session_cookie_name,
            timeout_s=settings.request_timeout_s,
        )
    owned_worker_client = worker_client is None
    if worker_client is None:
        # Kept as a late import so mock-only gateway tests do not initialize
        # signing material or a network client.
        from .worker_http_client import WorkerHttpClient

        worker_client = WorkerHttpClient.from_settings(settings)
    job_store = job_store or JobOwnershipStore(
        settings.job_db_path,
        retention_ttl_s=settings.job_ownership_ttl_s,
    )
    synchronizer = GatewaySynchronizer(
        novo_client=novo_client,
        worker_client=worker_client,
        settings=settings,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await job_store.initialize()
        yield
        if owned_novo_client and hasattr(novo_client, "aclose"):
            await novo_client.aclose()
        if owned_worker_client and hasattr(worker_client, "aclose"):
            await worker_client.aclose()

    app = FastAPI(
        title="Novo Chat gateway",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.novo_client = novo_client
    app.state.worker_client = worker_client
    app.state.job_store = job_store
    app.state.synchronizer = synchronizer

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(settings.base_path):
            response.headers["Cache-Control"] = "private, no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "same-origin"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
        return response

    @app.get(settings.base_path, include_in_schema=False)
    async def base_path_redirect() -> RedirectResponse:
        return RedirectResponse(f"{settings.base_path}/", status_code=status.HTTP_308_PERMANENT_REDIRECT)

    chat = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api = APIRouter(prefix="/api")

    async def session_and_context(request: Request) -> tuple[str, NovoContext]:
        session_value = request.cookies.get(settings.novo_session_cookie_name, "")
        if not session_value:
            raise NovoUnauthenticated
        context = await novo_client.context(session_value)
        return session_value, context

    @chat.exception_handler(NovoUnauthenticated)
    async def unauthenticated_handler(_request: Request, _exc: NovoUnauthenticated) -> JSONResponse:
        return JSONResponse(
            {"detail": "Novo sign-in is required", "loginUrl": login_url(settings)},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    @chat.exception_handler(NovoIntegrationUnavailable)
    async def novo_unavailable_handler(_request: Request, exc: NovoIntegrationUnavailable) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    @chat.exception_handler(WorkerUnavailable)
    async def worker_unavailable_handler(_request: Request, exc: WorkerUnavailable) -> JSONResponse:
        return JSONResponse(
            {
                "detail": "The compute service is unavailable or reconnecting.",
                "error": {"code": exc.code, "retryable": exc.retryable},
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @chat.exception_handler(WorkerRejected)
    async def worker_rejected_handler(_request: Request, exc: WorkerRejected) -> JSONResponse:
        return JSONResponse(
            {
                "detail": str(exc),
                "error": {"code": exc.code, "retryable": exc.retryable},
            },
            status_code=exc.status_code,
        )

    @chat.exception_handler(SynchronizationTimeout)
    async def sync_timeout_handler(_request: Request, _exc: SynchronizationTimeout) -> JSONResponse:
        return JSONResponse(
            {"detail": "Notebook preparation timed out. Try again."},
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )

    @chat.exception_handler(ExportLimitExceeded)
    async def export_limit_handler(_request: Request, _exc: ExportLimitExceeded) -> JSONResponse:
        return JSONResponse(
            {"detail": "This notebook is too large for the current Chat export limits."},
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    @chat.exception_handler(SynchronizationError)
    async def sync_error_handler(_request: Request, _exc: SynchronizationError) -> JSONResponse:
        return JSONResponse(
            {"detail": "Notebook preparation failed. Try again."},
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    @chat.get("/", response_class=HTMLResponse, response_model=None, include_in_schema=False)
    async def index(request: Request):
        try:
            await session_and_context(request)
        except NovoUnauthenticated:
            return RedirectResponse(login_url(settings), status_code=status.HTTP_303_SEE_OTHER)
        html = INDEX_HTML.read_text(encoding="utf-8").replace("__NOVO_CHAT_BASE_PATH__", settings.base_path)
        response = HTMLResponse(html)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'"
        )
        return response

    @chat.get("/static/app.js", include_in_schema=False)
    async def javascript() -> FileResponse:
        return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")

    @chat.get("/static/styles.css", include_in_schema=False)
    async def stylesheet() -> FileResponse:
        return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

    @chat.get("/healthz")
    async def healthz() -> dict[str, Any]:
        db_ok = await job_store.healthy()
        worker = await worker_availability(worker_client)
        return {
            "ok": db_ok,
            "gateway": "ok" if db_ok else "degraded",
            "worker": worker.model_dump(),
        }

    @api.get("/context")
    async def gateway_context(request: Request) -> dict[str, Any]:
        session_value, context = await session_and_context(request)
        worker = await worker_availability(worker_client)
        corpora = public_corpora(context)
        return {
            "apiVersion": "1",
            "user": context.user.model_dump(by_alias=True),
            "corpora": corpora,
            "worker": worker.model_dump(),
            "csrfToken": csrf_token(csrf_secret, session_value),
            "novoHomePath": settings.novo_home_path,
            "authorizationSource": "live Novo session and notebook permissions",
        }

    @api.get("/corpora")
    async def corpora(request: Request) -> dict[str, Any]:
        _session_value, context = await session_and_context(request)
        return {"corpora": public_corpora(context), "authorizationSource": "live Novo"}

    @api.get("/worker", response_model=WorkerAvailability)
    async def worker_status(request: Request) -> WorkerAvailability:
        await session_and_context(request)
        return await worker_availability(worker_client)

    @api.post("/jobs", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(request: Request) -> JobAccepted:
        session_value, context = await session_and_context(request)
        require_same_origin_json_csrf(request, settings, csrf_secret, session_value)
        body = await read_bounded_json_body(request, settings.gateway_max_request_bytes)
        try:
            submitted = GatewayJobRequest.model_validate_json(body)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body is invalid") from exc
        request_id = str(uuid.uuid4())
        idempotency_key = request.headers.get("idempotency-key") or request_id
        if not IDEMPOTENCY_KEY.fullmatch(idempotency_key):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Idempotency-Key")
        context, scope = await synchronized_scope(
            session_value=session_value,
            initial_context=context,
            submitted=submitted,
            settings=settings,
            novo_client=novo_client,
            synchronizer=synchronizer,
        )
        payload = submitted.model_dump(mode="json", exclude={"operation", "corpus"}, exclude_none=True)
        worker_operation = "query" if submitted.operation is GatewayOperation.ASK else submitted.operation.value
        worker_response = await worker_client.submit(
            operation=worker_operation,
            request_id=request_id,
            idempotency_key=idempotency_key,
            actor_user_id=context.user.id,
            scope=scope,
            payload=payload,
        )
        worker_job = unwrap_job(worker_response)
        job_id = str(worker_job.get("jobId") or "")
        if not job_id:
            raise WorkerRejected("Worker did not return a jobId")
        await job_store.add(job_id, context.user.id, worker_operation, scope)
        return JobAccepted(
            jobId=job_id,
            state=str(worker_job.get("state") or "queued"),
            operation=worker_operation,
        )

    @api.get("/jobs/{job_id}", response_model=JobPublicStatus)
    async def job_status(request: Request, job_id: str) -> JobPublicStatus:
        _context, _owned = await authorize_owned_job(
            request,
            job_id,
            session_and_context,
            job_store,
            settings.index_schema_version,
        )
        worker_response = await worker_client.job_status(job_id)
        public = public_job_status(worker_response)
        if public["jobId"] != job_id:
            raise WorkerRejected("Worker returned the wrong job")
        return JobPublicStatus.model_validate(public)

    @api.get("/jobs/{job_id}/result")
    async def job_result(request: Request, job_id: str) -> dict[str, Any]:
        context, owned = await authorize_owned_job(
            request,
            job_id,
            session_and_context,
            job_store,
            settings.index_schema_version,
        )
        worker_response = await worker_client.job_status(job_id)
        public = public_job_status(worker_response)
        if public["jobId"] != job_id:
            raise WorkerRejected("Worker returned the wrong job")
        result = completed_result(worker_response)
        # Fetch Novo context again after the worker response. A long or delayed
        # status request must not create a window where content is released
        # after its revision or the user's access changed.
        context, owned = await authorize_owned_job(
            request,
            job_id,
            session_and_context,
            job_store,
            settings.index_schema_version,
        )
        assert_result_scope(result, set(owned.notebook_ids), set(notebook_ids(context)))
        return {"jobId": job_id, "result": result}

    chat.include_router(api)
    app.mount(settings.base_path, chat, name="novo-chat")
    return app


async def worker_availability(worker_client: GatewayWorkerClient | Any) -> WorkerAvailability:
    try:
        health = await worker_client.health()
        explicit_ok = health.get("ok")
        protocol_ready = (
            str(health.get("state") or "").lower() == "ready"
            and health.get("queueHealthy", health.get("queue_healthy")) is True
            and health.get("indexServiceHealthy", health.get("index_service_healthy")) is True
        )
        healthy = bool(explicit_ok if explicit_ok is not None else health.get("status") == "ok" or protocol_ready)
        if not healthy:
            return WorkerAvailability(available=False, detail="The compute worker reported an unhealthy state.")
        capabilities, models = await asyncio.gather(worker_client.capabilities(), worker_client.model_status())
        return WorkerAvailability(
            available=True,
            detail="The compute service is available.",
            capabilities=capabilities,
            modelStatus=models.get("models", models),
        )
    except Exception:
        # Upstream connection details belong in service logs, not in a public
        # degraded-state response.
        return WorkerAvailability(
            available=False,
            detail="The compute service may be offline or reconnecting. Novo is still available.",
        )


def public_corpora(context: NovoContext) -> list[dict[str, Any]]:
    notebooks = sorted(context.notebooks, key=lambda row: (row.name.casefold(), row.id))
    result = [
        {
            "id": "all",
            "corpus_key": "novo:all",
            "name": "All my notebooks",
            "content_revision": aggregate_revision(notebooks),
            "notebook_count": len(notebooks),
        }
    ]
    result.extend(notebook.public_corpus() for notebook in notebooks)
    return result


def aggregate_revision(notebooks: list[Any]) -> str:
    material = "\n".join(f"{row.id}:{row.content_revision}" for row in notebooks).encode("utf-8")
    return f"sha256:{hashlib.sha256(material).hexdigest()}"


def notebook_ids(context: NovoContext) -> list[str]:
    return [notebook.id for notebook in context.notebooks]


def resolve_scope(
    context: NovoContext,
    submitted: GatewayJobRequest,
    index_schema_version: str,
) -> list[dict[str, str]]:
    if submitted.operation in {GatewayOperation.MODEL_START, GatewayOperation.MODEL_STOP}:
        return []
    by_id = {notebook.id: notebook for notebook in context.notebooks}
    if submitted.corpus == "novo:all":
        selected = list(by_id.values())
    elif submitted.corpus and submitted.corpus.startswith("novo:"):
        notebook_id = submitted.corpus.removeprefix("novo:")
        notebook = by_id.get(notebook_id)
        if not notebook:
            # Do not distinguish inaccessible from nonexistent notebooks.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")
        selected = [notebook]
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")
    if not selected:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No readable notebooks are available")
    return [
        {
            "notebookId": notebook.id,
            "contentRevision": notebook.content_revision,
            "indexSchemaVersion": index_schema_version,
        }
        for notebook in sorted(selected, key=lambda row: row.id)
    ]


async def authorize_owned_job(
    request: Request,
    job_id: str,
    context_loader: Any,
    job_store: JobOwnershipStore,
    index_schema_version: str,
) -> tuple[NovoContext, OwnedJob]:
    _session_value, context = await context_loader(request)
    owned = await job_store.get(job_id)
    if not owned or owned.user_id != context.user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    current_notebooks = set(notebook_ids(context))
    if not set(owned.notebook_ids).issubset(current_notebooks):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Notebook access changed before job release")
    if owned.scope is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Notebook content changed before job release. Submit a new request.",
        )
    current_by_id = {notebook.id: notebook.content_revision for notebook in context.notebooks}
    if any(
        current_by_id.get(item.notebook_id) != item.content_revision
        or item.index_schema_version != index_schema_version
        for item in owned.scope
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Notebook content changed before job release. Submit a new request.",
        )
    return context, owned


async def synchronized_scope(
    *,
    session_value: str,
    initial_context: NovoContext,
    submitted: GatewayJobRequest,
    settings: GatewaySettings,
    novo_client: NovoIntegrationClient | Any,
    synchronizer: GatewaySynchronizer,
) -> tuple[NovoContext, list[dict[str, str]]]:
    if submitted.operation in {GatewayOperation.MODEL_START, GatewayOperation.MODEL_STOP}:
        return initial_context, []

    context = initial_context
    for attempt in range(settings.sync_revision_retries + 1):
        scope = resolve_scope(context, submitted, settings.index_schema_version)
        try:
            if submitted.operation is GatewayOperation.ASK:
                await synchronizer.ensure_indexes_ready(
                    session_value=session_value,
                    actor_user_id=context.user.id,
                    scope=scope,
                )
            else:
                await synchronizer.prepare_documents_for_rebuild(
                    session_value=session_value,
                    actor_user_id=context.user.id,
                    scope=scope,
                )

            refreshed = await novo_client.context(session_value)
            if refreshed.user.id != context.user.id:
                raise NovoUnauthenticated
            refreshed_scope = resolve_scope(refreshed, submitted, settings.index_schema_version)
            if refreshed_scope != scope:
                raise NovoRevisionChanged(scope[0]["notebookId"], scope[0]["contentRevision"])
            return refreshed, refreshed_scope
        except (NovoRevisionChanged, NovoNotebookUnavailable):
            if attempt >= settings.sync_revision_retries:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Novo notebook content or access changed during preparation. Try again.",
                )
            context = await novo_client.context(session_value)

    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Notebook synchronization could not stabilize")


def assert_result_scope(result: Any, submitted_scope: set[str], current_scope: set[str]) -> None:
    allowed = submitted_scope & current_scope
    if not isinstance(result, dict):
        return
    rows: list[Any] = []
    for key in ("hits", "sources", "citations", "indexes"):
        value = result.get(key)
        if isinstance(value, list):
            rows.extend(value)
    for row in rows:
        if not isinstance(row, dict):
            continue
        notebook_id = row.get("notebookId") or row.get("notebook_id")
        if notebook_id and str(notebook_id) not in allowed:
            raise WorkerRejected("Worker result exceeded the authorized notebook scope")
        for key in ("sourceUrl", "source_url", "novoUrl", "novo_url"):
            source_url = row.get(key)
            if source_url is None:
                continue
            try:
                validate_same_origin_path(str(source_url))
            except ValueError as exc:
                raise WorkerRejected("Worker result contained an unsafe source URL") from exc
