from __future__ import annotations

from http.cookies import CookieError, SimpleCookie
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .secrets import FileSecret


class NovoUser(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    email: str
    first_name: str = Field(default="", alias="firstName")
    last_name: str = Field(default="", alias="lastName")
    display_name: str = Field(default="", alias="displayName")
    role: str = "member"

    @property
    def label(self) -> str:
        return self.display_name or " ".join(part for part in (self.first_name, self.last_name) if part) or self.email


class NovoNotebook(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str
    access_role: str = Field(default="viewer", alias="accessRole")
    content_revision: str = Field(alias="contentRevision")
    updated_at: str | None = Field(default=None, alias="updatedAt")
    page_count: int = Field(default=0, alias="pageCount")
    attachment_count: int = Field(default=0, alias="attachmentCount")
    text_chars: int = Field(default=0, alias="textChars")

    def public_corpus(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "corpus_key": f"novo:{self.id}",
            "name": self.name,
            "access_role": self.access_role,
            "content_revision": self.content_revision,
            "updated_at": self.updated_at,
            "page_count": self.page_count,
            "attachment_count": self.attachment_count,
            "text_chars": self.text_chars,
        }


class NovoContext(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    api_version: str = Field(default="1", alias="apiVersion")
    novo_version: str = Field(default="", alias="novoVersion")
    user: NovoUser
    notebooks: list[NovoNotebook] = Field(default_factory=list)


class NovoExportAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    name: str
    mime_type: str = Field(default="application/octet-stream", alias="mimeType")
    size: int = Field(default=0, ge=0)
    block_type: str | None = Field(default=None, alias="blockType")
    created_at: str | None = Field(default=None, alias="createdAt")

    def worker_document(self) -> dict[str, Any]:
        return {
            "attachmentId": self.id,
            "name": self.name,
            "mimeType": self.mime_type,
            "size": self.size,
            "blockType": self.block_type,
            "createdAt": self.created_at,
        }


class NovoExportPage(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    title: str = ""
    text: str = ""
    status: str | None = None
    created_at: str | None = Field(default=None, alias="createdAt")
    updated_at: str | None = Field(default=None, alias="updatedAt")
    tags: list[str] = Field(default_factory=list)
    attachments: list[NovoExportAttachment] = Field(default_factory=list)
    source_url: str = Field(alias="sourceUrl")

    def worker_document(self) -> dict[str, Any]:
        # Novo's general-purpose ``updatedAt`` also changes for non-indexable
        # UI state such as page locking and attachment annotations.  Do not put
        # it into the revision-keyed derived document: the same contentRevision
        # must always produce the same canonical worker payload.
        return {
            "pageId": self.id,
            "title": self.title,
            "text": self.text,
            "status": self.status,
            "createdAt": self.created_at,
            "tags": self.tags,
            "attachments": [attachment.worker_document() for attachment in self.attachments],
            "sourceUrl": self.source_url,
        }


class NovoExportNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    name: str
    content_revision: str = Field(alias="contentRevision")


class NovoPagesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    api_version: str = Field(default="1", alias="apiVersion")
    notebook: NovoExportNotebook
    pages: list[NovoExportPage]
    next_cursor: str | None = Field(default=None, alias="nextCursor", max_length=2048)
    complete: bool


class NovoUnauthenticated(Exception):
    pass


class NovoIntegrationUnavailable(Exception):
    pass


class NovoRevisionChanged(Exception):
    """The requested export revision changed before it could be completed."""

    def __init__(self, notebook_id: str, expected_revision: str) -> None:
        super().__init__("Novo notebook content changed during synchronization")
        self.notebook_id = notebook_id
        self.expected_revision = expected_revision


class NovoNotebookUnavailable(Exception):
    """An export target is nonexistent or no longer readable."""

    pass


class NovoIntegrationClient:
    """Client for the loopback-only Novo integration API.

    The browser's single ``eln_session`` value is attached only here. No method
    on the worker client accepts cookies or a raw browser request.
    """

    def __init__(
        self,
        *,
        base_url: str,
        secret: FileSecret,
        session_cookie_name: str,
        timeout_s: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.secret = secret
        self.session_cookie_name = session_cookie_name
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, trust_env=False)

    async def context(self, session_value: str) -> NovoContext:
        response = await self._get(
            f"{self.base_url}/context",
            session_value=session_value,
        )
        if response.status_code in {401, 403}:
            raise NovoUnauthenticated
        if response.status_code != 200:
            raise NovoIntegrationUnavailable(f"Novo integration API returned HTTP {response.status_code}")
        try:
            return NovoContext.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise NovoIntegrationUnavailable("Novo integration API returned an invalid context") from exc

    async def export_pages(
        self,
        session_value: str,
        *,
        notebook_id: str,
        content_revision: str,
        cursor: str | None = None,
        limit: int = 8,
    ) -> NovoPagesResponse:
        if not 1 <= limit <= 250:
            raise ValueError("Novo export page limit must be between 1 and 250")
        if not content_revision or any(character in content_revision for character in {'"', "\\", "\r", "\n"}):
            raise NovoIntegrationUnavailable("Novo returned an invalid content revision")
        response = await self._get(
            f"{self.base_url}/notebooks/{quote(notebook_id, safe='-._~')}/pages",
            session_value=session_value,
            headers={"If-Match": f'"{content_revision}"'},
            params={"limit": str(limit), **({"cursor": cursor} if cursor else {})},
        )
        if response.status_code == 412:
            raise NovoRevisionChanged(notebook_id, content_revision)
        if response.status_code in {401, 403}:
            raise NovoUnauthenticated
        if response.status_code == 404:
            raise NovoNotebookUnavailable
        if response.status_code != 200:
            raise NovoIntegrationUnavailable(f"Novo page export returned HTTP {response.status_code}")
        try:
            page = NovoPagesResponse.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise NovoIntegrationUnavailable("Novo page export returned an invalid response") from exc
        if page.notebook.id != notebook_id or page.notebook.content_revision != content_revision:
            raise NovoRevisionChanged(notebook_id, content_revision)
        if len(page.pages) > limit:
            raise NovoIntegrationUnavailable("Novo page export exceeded the requested page limit")
        if page.complete and page.next_cursor:
            raise NovoIntegrationUnavailable("Completed Novo export returned another cursor")
        if not page.complete and not page.next_cursor:
            raise NovoIntegrationUnavailable("Incomplete Novo export omitted its next cursor")
        return page

    async def _get(
        self,
        url: str,
        *,
        session_value: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        cookie = SimpleCookie()
        try:
            cookie[self.session_cookie_name] = session_value
            cookie_header = cookie.output(header="").strip()
        except CookieError as exc:
            raise NovoUnauthenticated from exc
        try:
            response = await self._client.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.secret.read_text()}",
                    "Cookie": cookie_header,
                    **(headers or {}),
                },
                params=params,
            )
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            raise NovoIntegrationUnavailable("Novo session service is unavailable") from exc
        return response

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
