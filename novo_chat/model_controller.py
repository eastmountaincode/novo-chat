"""Unprivileged worker boundary for an allowlisted model controller.

The privileged Unix-socket implementation intentionally lives outside this
module.  The worker accepts a small adapter and never receives Docker commands,
images, arguments, ports, or filesystem paths.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Mapping, Protocol

from .protocol import ModelRuntimeState


class ModelControllerError(Exception):
    """Safe adapter failure; its message must not contain backend details."""

    def __init__(
        self,
        code: str = "MODEL_CONTROLLER_UNAVAILABLE",
        message: str = "Model control is unavailable.",
        *,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ModelController(Protocol):
    """The only model-control surface available to the unprivileged worker."""

    def status(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        ...

    def start(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        ...

    def stop(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        ...


class ModelReadinessProbe(Protocol):
    def is_ready(self, model: str) -> bool:
        ...

    def wait_until_ready(self, model: str, *, timeout_seconds: float) -> bool:
        ...


class UnavailableModelReadinessProbe:
    """Fail-closed default when no configured HTTP backend was injected."""

    def is_ready(self, model: str) -> bool:
        del model
        return False

    def wait_until_ready(self, model: str, *, timeout_seconds: float) -> bool:
        del model, timeout_seconds
        return False


class UnavailableModelController:
    """Safe default used when no local controller adapter is configured."""

    def status(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        return {"state": ModelRuntimeState.UNAVAILABLE.value}

    def start(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        raise ModelControllerError()

    def stop(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        raise ModelControllerError()


def model_state_from_response(response: Mapping[str, Any]) -> ModelRuntimeState:
    """Discard all controller fields except the typed public state."""

    raw_state = str(response.get("state", ModelRuntimeState.UNAVAILABLE.value)).lower()
    if raw_state == "running":
        raw_state = ModelRuntimeState.READY.value
    try:
        return ModelRuntimeState(raw_state)
    except ValueError:
        return ModelRuntimeState.FAILED


_SAFE_CONTROLLER_ERRORS = {
    "MODEL_CONFLICT": "Another approved model is currently active.",
    "MODEL_INVENTORY_UNAVAILABLE": "Model control is unavailable.",
    "MODEL_LOCK_UNAVAILABLE": "Model control is unavailable.",
    "UNKNOWN_MODEL": "Requested model is not approved.",
    "CONTAINER_UNAVAILABLE": "Configured model container is unavailable.",
    "DOCKER_UNAVAILABLE": "Model control is unavailable.",
    "DOCKER_TIMEOUT": "Model control timed out.",
    "MODEL_START_FAILED": "Model could not be started.",
    "MODEL_STOP_FAILED": "Model could not be stopped.",
    "REQUEST_TIMEOUT": "Model control timed out.",
}


class UnixSocketModelController:
    """Typed client for deploy/compute-host/modelctl_server.py's JSON-line socket."""

    def __init__(self, socket_path: str | Path, *, timeout_seconds: float = 10.0, max_response_bytes: int = 65536):
        path = Path(socket_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("model controller socket path must be absolute")
        if timeout_seconds <= 0 or max_response_bytes < 1024:
            raise ValueError("invalid model controller client limits")
        self.socket_path = path
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)

    def status(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        return self._call("status", model, request_id=request_id, actor_user_id=actor_user_id)

    def start(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        return self._call("start", model, request_id=request_id, actor_user_id=actor_user_id)

    def stop(self, model: str, *, request_id: str, actor_user_id: str) -> Mapping[str, Any]:
        return self._call("stop", model, request_id=request_id, actor_user_id=actor_user_id)

    def _call(
        self,
        action: str,
        model: str,
        *,
        request_id: str,
        actor_user_id: str,
    ) -> Mapping[str, Any]:
        if action not in {"status", "start", "stop"}:
            raise ModelControllerError()
        request = {
            "version": 1,
            "requestId": request_id,
            "actorId": actor_user_id,
            "action": action,
            "model": model,
        }
        wire = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(wire) > 8192:
            raise ModelControllerError("MODEL_CONTROLLER_REQUEST_INVALID", "Model control request is invalid.")
        chunks: list[bytes] = []
        size = 0
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.connect(os.fspath(self.socket_path))
                connection.sendall(wire)
                while True:
                    chunk = connection.recv(min(4096, self.max_response_bytes + 1 - size))
                    if not chunk:
                        break
                    newline = chunk.find(b"\n")
                    if newline >= 0:
                        chunks.append(chunk[:newline])
                        size += newline
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise ModelControllerError(
                            "MODEL_CONTROLLER_RESPONSE_INVALID",
                            "Model controller returned an invalid response.",
                        )
        except ModelControllerError:
            raise
        except (OSError, socket.timeout) as exc:
            raise ModelControllerError() from exc
        if size > self.max_response_bytes:
            raise ModelControllerError(
                "MODEL_CONTROLLER_RESPONSE_INVALID",
                "Model controller returned an invalid response.",
            )
        try:
            response = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelControllerError(
                "MODEL_CONTROLLER_RESPONSE_INVALID",
                "Model controller returned an invalid response.",
            ) from exc
        if not isinstance(response, dict) or response.get("version") != 1:
            raise ModelControllerError(
                "MODEL_CONTROLLER_RESPONSE_INVALID",
                "Model controller returned an invalid response.",
            )
        if response.get("requestId") != request_id:
            raise ModelControllerError(
                "MODEL_CONTROLLER_RESPONSE_INVALID",
                "Model controller returned an invalid response.",
            )
        if response.get("ok") is not True:
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            code = str(error.get("code") or "MODEL_CONTROLLER_UNAVAILABLE")
            message = _SAFE_CONTROLLER_ERRORS.get(code, "Model control is unavailable.")
            raise ModelControllerError(code if code in _SAFE_CONTROLLER_ERRORS else "MODEL_CONTROLLER_UNAVAILABLE", message)
        if response.get("model") != model or response.get("action") != action or not isinstance(response.get("status"), dict):
            raise ModelControllerError(
                "MODEL_CONTROLLER_RESPONSE_INVALID",
                "Model controller returned an invalid response.",
            )
        status = response["status"]
        state = str(status.get("state") or "")
        if state not in {"running", "stopped", "starting", "stopping", "failed"}:
            raise ModelControllerError(
                "MODEL_CONTROLLER_RESPONSE_INVALID",
                "Model controller returned an invalid response.",
            )
        return {"state": state, "changed": bool(status.get("changed", False))}


__all__ = [
    "ModelController",
    "ModelControllerError",
    "ModelReadinessProbe",
    "UnavailableModelReadinessProbe",
    "UnavailableModelController",
    "UnixSocketModelController",
    "model_state_from_response",
]
