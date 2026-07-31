from __future__ import annotations

import os
from pathlib import Path


class FileSecret:
    """Read a service secret from a root-managed file and cache by mtime."""

    def __init__(self, path: Path, *, minimum_bytes: int = 16) -> None:
        self.path = path
        self.minimum_bytes = minimum_bytes
        self._mtime_ns: int | None = None
        self._value: bytes | None = None

    def read_bytes(self) -> bytes:
        stat = self.path.stat()
        if stat.st_mode & 0o007:
            raise RuntimeError(f"secret file must not be accessible to other users: {self.path}")
        if self._value is None or stat.st_mtime_ns != self._mtime_ns:
            value = self.path.read_bytes().strip()
            if len(value) < self.minimum_bytes:
                raise RuntimeError(f"secret file is empty or too short: {self.path}")
            self._value = value
            self._mtime_ns = stat.st_mtime_ns
        return self._value

    def read_text(self) -> str:
        try:
            return self.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"secret file is not UTF-8 text: {self.path}") from exc


def write_test_secret(path: Path, value: str) -> None:
    """Test helper kept here so fixtures create production-equivalent files."""

    path.write_text(value, encoding="utf-8")
    os.chmod(path, 0o600)
