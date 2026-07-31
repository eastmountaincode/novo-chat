#!/usr/bin/env python3
"""Root-owned, allowlisted Docker controller for Novo Chat.

This file intentionally supports Python 3.8. It accepts JSON-line requests only through a Unix socket supplied by
systemd socket activation. It never opens a network listener or invokes a shell.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
from typing import Any, Dict, Optional, Tuple


PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 8192
HOST_LOCK_PATH = "/run/novo-chat/modelctl-gpu.lock"
CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
ACTOR_ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
ALLOWED_FIELDS = {"version", "requestId", "actorId", "action", "model"}


class ControllerError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def load_inventory(path: str) -> Tuple[str, ...]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("version") != PROTOCOL_VERSION:
        raise RuntimeError("model controller inventory must use version 1")
    containers = data.get("containers")
    if not isinstance(containers, list) or not containers:
        raise RuntimeError("model controller inventory must define containers")
    normalized = []
    seen = set()
    for container in containers:
        if not isinstance(container, str) or not CONTAINER_RE.fullmatch(container):
            raise RuntimeError("invalid container in model controller inventory")
        if container in seen:
            raise RuntimeError("duplicate container in model controller inventory")
        seen.add(container)
        normalized.append(container)
    return tuple(normalized)


def load_config(path: str, inventory_path: str) -> Dict[str, Any]:
    exclusive_containers = load_inventory(inventory_path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("version") != PROTOCOL_VERSION:
        raise RuntimeError("model controller config must use version 1")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError("model controller config must define models")
    normalized = {}
    for model, raw in models.items():
        if not isinstance(model, str) or not MODEL_ID_RE.fullmatch(model) or not isinstance(raw, dict):
            raise RuntimeError("invalid model configuration")
        container = raw.get("container")
        if not isinstance(container, str) or not CONTAINER_RE.fullmatch(container):
            raise RuntimeError("invalid container for model: %s" % model)
        if container not in exclusive_containers:
            raise RuntimeError("model container is absent from the shared inventory: %s" % model)
        stop_timeout = int(raw.get("stopTimeoutSeconds", 30))
        if stop_timeout < 1 or stop_timeout > 120:
            raise RuntimeError("invalid stop timeout for model: %s" % model)
        normalized[model] = {
            "container": container,
            "stopTimeoutSeconds": stop_timeout,
        }
    return {
        "version": PROTOCOL_VERSION,
        "dockerBinary": str(data.get("dockerBinary") or "/usr/bin/docker"),
        "models": normalized,
        "inventoryPath": os.path.abspath(inventory_path),
    }


class DockerController:
    def __init__(self, config: Dict[str, Any], *, lock_path: str = HOST_LOCK_PATH):
        self.config = config
        self.lock_path = lock_path
        self._lock = threading.Lock()

    @contextmanager
    def _host_lock(self):
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
            ):
                raise OSError("unsafe model controller lock file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except OSError:
            raise ControllerError(
                "MODEL_LOCK_UNAVAILABLE",
                "Model exclusivity control is unavailable.",
                503,
            )
        finally:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)

    def _run(self, args, timeout: int = 60) -> subprocess.CompletedProcess:
        command = [self.config["dockerBinary"]] + list(args)
        try:
            return subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
            )
        except subprocess.TimeoutExpired:
            raise ControllerError("DOCKER_TIMEOUT", "Container operation timed out.", 503)
        except OSError:
            raise ControllerError("DOCKER_UNAVAILABLE", "Container controller is unavailable.", 503)

    def _is_running(self, container: str) -> bool:
        result = self._run(["inspect", "--format", "{{.State.Running}}", container], timeout=15)
        if result.returncode != 0:
            raise ControllerError("CONTAINER_UNAVAILABLE", "Configured model container is unavailable.", 503)
        return result.stdout.strip().lower() == "true"

    def _exclusive_containers(self) -> Tuple[str, ...]:
        try:
            containers = load_inventory(self.config["inventoryPath"])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            raise ControllerError(
                "MODEL_INVENTORY_UNAVAILABLE",
                "Model exclusivity inventory is unavailable.",
                503,
            )
        configured = {entry["container"] for entry in self.config["models"].values()}
        if not configured.issubset(containers):
            raise ControllerError(
                "MODEL_INVENTORY_UNAVAILABLE",
                "Model exclusivity inventory is unavailable.",
                503,
            )
        return containers

    def status(self, model: str) -> Dict[str, Any]:
        config = self._model(model)
        # The privileged boundary reports only container runtime state.  The
        # unprivileged worker separately probes the model HTTP endpoint before
        # exposing the public semantic state as ``ready``.
        return {"state": "running" if self._is_running(config["container"]) else "stopped"}

    def start(self, model: str) -> Dict[str, Any]:
        config = self._model(model)
        with self._lock:
            with self._host_lock():
                running_containers = [
                    container
                    for container in self._exclusive_containers()
                    if self._is_running(container)
                ]
                if any(container != config["container"] for container in running_containers):
                    raise ControllerError(
                        "MODEL_CONFLICT",
                        "Another approved model is running; drain and stop it before switching.",
                        409,
                    )
                if config["container"] in running_containers:
                    return {"state": "running", "changed": False}
                result = self._run(["start", config["container"]], timeout=60)
                if result.returncode != 0:
                    raise ControllerError("MODEL_START_FAILED", "Model container could not be started.", 503)
                return {"state": "running", "changed": True}

    def stop(self, model: str) -> Dict[str, Any]:
        config = self._model(model)
        with self._lock:
            with self._host_lock():
                if not self._is_running(config["container"]):
                    return {"state": "stopped", "changed": False}
                result = self._run(
                    ["stop", "--time", str(config["stopTimeoutSeconds"]), config["container"]],
                    timeout=config["stopTimeoutSeconds"] + 15,
                )
                if result.returncode != 0:
                    raise ControllerError("MODEL_STOP_FAILED", "Model container could not be stopped.", 503)
                return {"state": "stopped", "changed": True}

    def _model(self, model: str) -> Dict[str, Any]:
        config = self.config["models"].get(model)
        if not config:
            raise ControllerError("UNKNOWN_MODEL", "Model is not approved.", 404)
        return config


def parse_request(raw: bytes) -> Dict[str, Any]:
    try:
        request = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ControllerError("INVALID_JSON", "Request must be valid UTF-8 JSON.")
    if not isinstance(request, dict):
        raise ControllerError("INVALID_REQUEST", "Request must be a JSON object.")
    if set(request) - ALLOWED_FIELDS:
        raise ControllerError("INVALID_REQUEST", "Request contains unsupported fields.")
    if request.get("version") != PROTOCOL_VERSION:
        raise ControllerError("UNSUPPORTED_VERSION", "Unsupported controller protocol version.")
    request_id = request.get("requestId")
    if not isinstance(request_id, str) or not REQUEST_ID_RE.fullmatch(request_id):
        raise ControllerError("INVALID_REQUEST_ID", "A valid request ID is required.")
    actor_id = request.get("actorId")
    if not isinstance(actor_id, str) or not ACTOR_ID_RE.fullmatch(actor_id):
        raise ControllerError("INVALID_ACTOR", "A valid actor ID is required.")
    action = request.get("action")
    if action not in {"status", "start", "stop"}:
        raise ControllerError("UNKNOWN_ACTION", "Action is not approved.")
    model = request.get("model")
    if not isinstance(model, str) or not MODEL_ID_RE.fullmatch(model):
        raise ControllerError("INVALID_MODEL", "A valid model ID is required.")
    return request


def handle_request(controller: DockerController, raw: bytes) -> Tuple[Dict[str, Any], int]:
    request_id = ""
    try:
        request = parse_request(raw)
        request_id = request["requestId"]
        action = request["action"]
        model = request["model"]
        actor_id = request["actorId"]
        print(
            "modelctl requestId=%s actorId=%s action=%s model=%s"
            % (request_id, actor_id, action, model),
            flush=True,
        )
        if action == "status":
            status = controller.status(model)
        elif action == "start":
            status = controller.start(model)
        else:
            status = controller.stop(model)
        return {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "requestId": request_id,
            "model": model,
            "action": action,
            "status": status,
        }, 200
    except ControllerError as exc:
        return {
            "ok": False,
            "version": PROTOCOL_VERSION,
            "requestId": request_id,
            "error": {"code": exc.code, "message": exc.message},
        }, exc.status
    except Exception:
        return {
            "ok": False,
            "version": PROTOCOL_VERSION,
            "requestId": request_id,
            "error": {"code": "INTERNAL_ERROR", "message": "Model controller failed."},
        }, 500


def read_request(connection: socket.socket) -> bytes:
    chunks = []
    size = 0
    while True:
        chunk = connection.recv(min(4096, MAX_REQUEST_BYTES + 1 - size))
        if not chunk:
            break
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunk = chunk[:newline]
            chunks.append(chunk)
            size += len(chunk)
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise ControllerError("REQUEST_TOO_LARGE", "Request is too large.", 413)
    if size > MAX_REQUEST_BYTES:
        raise ControllerError("REQUEST_TOO_LARGE", "Request is too large.", 413)
    return b"".join(chunks)


def activated_socket() -> Optional[socket.socket]:
    if os.environ.get("LISTEN_PID") != str(os.getpid()):
        return None
    if int(os.environ.get("LISTEN_FDS", "0")) != 1:
        return None
    return socket.fromfd(3, socket.AF_UNIX, socket.SOCK_STREAM)


def create_socket(path: str) -> socket.socket:
    if os.path.exists(path):
        raise RuntimeError("refusing to replace an existing socket path")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    os.chmod(path, 0o660)
    listener.listen(64)
    return listener


def serve(listener: socket.socket, controller: DockerController) -> None:
    stopping = threading.Event()

    def stop(_signum, _frame):
        stopping.set()
        try:
            listener.close()
        except OSError:
            pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping.is_set():
        try:
            connection, _ = listener.accept()
        except OSError:
            if stopping.is_set():
                break
            raise
        with connection:
            connection.settimeout(5)
            try:
                raw = read_request(connection)
                payload, status = handle_request(controller, raw)
            except socket.timeout:
                payload = {
                    "ok": False,
                    "version": PROTOCOL_VERSION,
                    "requestId": "",
                    "error": {"code": "REQUEST_TIMEOUT", "message": "Request timed out."},
                }
                status = 408
            except ControllerError as exc:
                payload = {
                    "ok": False,
                    "version": PROTOCOL_VERSION,
                    "requestId": "",
                    "error": {"code": exc.code, "message": exc.message},
                }
                status = exc.status
            payload["statusCode"] = status
            connection.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--socket", help="Development fallback; production uses systemd socket activation")
    args = parser.parse_args()
    config = load_config(args.config, args.inventory)
    listener = activated_socket()
    if listener is None:
        if not args.socket:
            print("model controller requires one systemd socket", file=sys.stderr)
            return 2
        listener = create_socket(args.socket)
    serve(listener, DockerController(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
