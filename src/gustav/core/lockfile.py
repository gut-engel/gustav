"""Single-instance lock.

An exclusive ``flock`` on a PID file. The kernel drops the lock when the process
dies, so a crashed daemon never leaves a lock that needs manual cleaning, while a
*live* daemon cannot be started twice.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class LockHeld(RuntimeError):
    def __init__(self, path: Path, info: dict[str, Any] | None):
        pid = (info or {}).get("pid", "unknown")
        super().__init__(
            f"another gustav daemon is already running (pid {pid}, lock {path}). "
            f"Use 'ai status' to talk to it, or stop it first."
        )
        self.path = path
        self.info = info or {}


class PidLock:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self, info: dict[str, Any] | None = None) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LockHeld(self.path, self.read()) from exc
            raise
        payload = {
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            **(info or {}),
        }
        os.lseek(fd, 0, os.SEEK_SET)
        os.truncate(fd, 0)
        os.write(fd, (json.dumps(payload) + "\n").encode())
        os.fsync(fd)
        self._fd = fd
        return payload

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        finally:
            self._fd = None
        # Best effort: remove the file so a stale PID is not reported by `ai status`.
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def read(self) -> dict[str, Any] | None:
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, NotADirectoryError):
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def is_locked(self) -> bool:
        """True if some *live* process holds the lock."""
        if not self.path.exists():
            return False
        try:
            fd = os.open(self.path, os.O_RDWR)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def __enter__(self) -> "PidLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
