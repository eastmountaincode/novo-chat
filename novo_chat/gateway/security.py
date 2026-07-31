from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlencode, urlsplit

from fastapi import HTTPException, Request, status

from .config import GatewaySettings
from .secrets import FileSecret


def login_url(settings: GatewaySettings) -> str:
    return_to = f"{settings.base_path}/"
    separator = "&" if "?" in settings.novo_login_path else "?"
    return f"{settings.novo_login_path}{separator}{urlencode({settings.login_return_parameter: return_to})}"


def csrf_token(secret: FileSecret, session_value: str) -> str:
    return hmac.new(secret.read_bytes(), session_value.encode("utf-8"), hashlib.sha256).hexdigest()


def _request_origin(request: Request, settings: GatewaySettings) -> str:
    if settings.public_origin:
        return settings.public_origin
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "")).split(",", 1)[0].strip()
    return f"{scheme}://{host}"


def require_same_origin_json_csrf(
    request: Request,
    settings: GatewaySettings,
    secret: FileSecret,
    session_value: str,
) -> None:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="JSON is required")

    origin = request.headers.get("origin")
    if not origin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="A same-origin request is required")
    supplied_origin = urlsplit(origin)
    expected_origin = urlsplit(_request_origin(request, settings))
    if (supplied_origin.scheme, supplied_origin.netloc) != (expected_origin.scheme, expected_origin.netloc):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-origin request rejected")

    supplied = request.headers.get("x-csrf-token", "")
    expected = csrf_token(secret, session_value)
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")
