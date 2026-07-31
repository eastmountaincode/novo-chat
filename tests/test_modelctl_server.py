from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import queue
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "deploy" / "compute-host" / "modelctl_server.py"
SPEC = importlib.util.spec_from_file_location("modelctl_server", MODULE_PATH)
assert SPEC and SPEC.loader
modelctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(modelctl)


class SharedStateDockerController(modelctl.DockerController):
    """Filesystem-backed Docker stand-in shared by forked test processes."""

    def __init__(self, config, *, lock_path: str, state_path: str):
        super().__init__(config, lock_path=lock_path)
        self.state_path = Path(state_path)

    def _is_running(self, container: str) -> bool:
        running = self.state_path.read_text(encoding="utf-8").strip()
        if not running:
            # Make the unprotected read/check/start race deterministic enough
            # that the regression proves the host lock, not process timing.
            time.sleep(0.05)
        return running == container

    def _run(self, args, timeout: int = 60) -> subprocess.CompletedProcess:
        del timeout
        if args[0] == "start":
            self.state_path.write_text(args[1], encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError("unexpected fake Docker operation: %r" % (args,))


def start_in_process(
    config_path: str,
    inventory_path: str,
    lock_path: str,
    state_path: str,
    model: str,
    barrier,
    results,
) -> None:
    try:
        config = modelctl.load_config(config_path, inventory_path)
        controller = SharedStateDockerController(
            config,
            lock_path=lock_path,
            state_path=state_path,
        )
        barrier.wait(timeout=5)
        status = controller.start(model)
        results.put(("ok", status["changed"]))
    except modelctl.ControllerError as exc:
        results.put(("controller-error", exc.code))
    except Exception as exc:  # pragma: no cover - diagnostic for child failure
        results.put(("unexpected", repr(exc)))


class FakeController:
    def status(self, model: str):
        if model != "small":
            raise modelctl.ControllerError("UNKNOWN_MODEL", "Model is not approved.", 404)
        return {"state": "stopped"}

    def start(self, model: str):
        if model != "small":
            raise modelctl.ControllerError("UNKNOWN_MODEL", "Model is not approved.", 404)
        return {"state": "running", "changed": True}

    def stop(self, model: str):
        if model != "small":
            raise modelctl.ControllerError("UNKNOWN_MODEL", "Model is not approved.", 404)
        return {"state": "stopped", "changed": True}


def request(**overrides):
    payload = {
        "version": 1,
        "requestId": "req-1",
        "actorId": "user-1",
        "action": "status",
        "model": "small",
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


def write_inventory(path: Path, *containers: str) -> None:
    path.write_text(
        json.dumps({"version": 1, "containers": list(containers)}),
        encoding="utf-8",
    )


def write_config(path: Path, model: str, container: str) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    model: {
                        "container": container,
                        "stopTimeoutSeconds": 30,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class ModelControllerProtocolTests(unittest.TestCase):
    def test_status(self):
        payload, status = modelctl.handle_request(FakeController(), request())
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"]["state"], "stopped")

    def test_start_reports_container_runtime_state(self):
        payload, status = modelctl.handle_request(FakeController(), request(action="start"))
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"]["state"], "running")

    def test_unknown_action_is_rejected(self):
        payload, status = modelctl.handle_request(FakeController(), request(action="exec"))
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "UNKNOWN_ACTION")

    def test_extra_command_field_is_rejected(self):
        payload, status = modelctl.handle_request(FakeController(), request(command="docker rm -f anything"))
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "INVALID_REQUEST")

    def test_unknown_model_is_sanitized(self):
        payload, status = modelctl.handle_request(FakeController(), request(model="not-approved", action="start"))
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "UNKNOWN_MODEL")

    def test_model_id_uses_protocol_safe_token_alphabet(self):
        payload, status = modelctl.handle_request(FakeController(), request(model="unsafe+model"))
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "INVALID_MODEL")

    def test_config_rejects_unsafe_container(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "models.json"
            inventory_path = Path(temp_dir) / "inventory.json"
            write_inventory(inventory_path, "safe-container")
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "models": {"bad": {"container": "name; touch /tmp/pwned"}},
                    }
                )
            )
            with self.assertRaises(RuntimeError):
                modelctl.load_config(str(path), str(inventory_path))

    def test_config_container_must_be_in_shared_inventory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "models.json"
            inventory_path = Path(temp_dir) / "inventory.json"
            write_config(path, "small", "staging-model")
            write_inventory(inventory_path, "production-model")
            with self.assertRaisesRegex(RuntimeError, "absent from the shared inventory"):
                modelctl.load_config(str(path), str(inventory_path))

    def test_inventory_rejects_duplicate_containers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inventory_path = Path(temp_dir) / "inventory.json"
            write_inventory(inventory_path, "same-model", "same-model")
            with self.assertRaisesRegex(RuntimeError, "duplicate container"):
                modelctl.load_inventory(str(inventory_path))

    def test_start_reloads_shared_inventory_while_holding_host_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "models-staging.json"
            inventory_path = root / "inventory.json"
            state_path = root / "running-container"
            write_config(config_path, "staging", "staging-model")
            write_inventory(inventory_path, "staging-model")
            config = modelctl.load_config(str(config_path), str(inventory_path))

            write_inventory(inventory_path, "staging-model", "production-model")
            state_path.write_text("production-model", encoding="utf-8")
            controller = SharedStateDockerController(
                config,
                lock_path=str(root / "modelctl-gpu.lock"),
                state_path=str(state_path),
            )

            with self.assertRaises(modelctl.ControllerError) as raised:
                controller.start("staging")
            self.assertEqual(raised.exception.code, "MODEL_CONFLICT")

    def test_start_fails_closed_when_host_lock_cannot_be_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "models.json"
            inventory_path = root / "inventory.json"
            state_path = root / "running-container"
            write_config(config_path, "small", "small-model")
            write_inventory(inventory_path, "small-model")
            state_path.write_text("", encoding="utf-8")
            controller = SharedStateDockerController(
                modelctl.load_config(str(config_path), str(inventory_path)),
                lock_path=str(root / "missing" / "modelctl-gpu.lock"),
                state_path=str(state_path),
            )

            with self.assertRaises(modelctl.ControllerError) as raised:
                controller.start("small")
            self.assertEqual(raised.exception.code, "MODEL_LOCK_UNAVAILABLE")
            self.assertEqual(raised.exception.status, 503)

    @unittest.skipUnless(hasattr(os, "fork"), "cross-process flock test requires fork")
    def test_shared_lock_prevents_cross_config_model_start_race(self):
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_path = root / "models-staging.json"
            production_path = root / "models-production.json"
            inventory_path = root / "inventory.json"
            lock_path = root / "modelctl-gpu.lock"
            state_path = root / "running-container"
            write_config(staging_path, "staging", "staging-model")
            write_config(production_path, "production", "production-model")
            write_inventory(inventory_path, "staging-model", "production-model")
            state_path.write_text("", encoding="utf-8")

            barrier = context.Barrier(2)
            results = context.Queue()
            processes = [
                context.Process(
                    target=start_in_process,
                    args=(
                        str(config_path),
                        str(inventory_path),
                        str(lock_path),
                        str(state_path),
                        model,
                        barrier,
                        results,
                    ),
                )
                for config_path, model in (
                    (staging_path, "staging"),
                    (production_path, "production"),
                )
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                    self.fail("cross-process model start test hung")
                self.assertEqual(process.exitcode, 0)

            outcomes = []
            try:
                for _ in processes:
                    outcomes.append(results.get(timeout=2))
            except queue.Empty:
                self.fail("child process did not report a model start outcome")
            finally:
                results.close()

            self.assertCountEqual(
                outcomes,
                [("ok", True), ("controller-error", "MODEL_CONFLICT")],
            )
            self.assertIn(
                state_path.read_text(encoding="utf-8"),
                {"staging-model", "production-model"},
            )


if __name__ == "__main__":
    unittest.main()
