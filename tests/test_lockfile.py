from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from gustav.core.lockfile import LockHeld, PidLock


def test_acquire_writes_pid_and_info(tmp_path):
    lock = PidLock(tmp_path / "gustav.pid")
    info = lock.acquire({"socket": "/tmp/x.sock"})
    try:
        assert info["pid"] == os.getpid()
        assert lock.read()["socket"] == "/tmp/x.sock"
        assert lock.is_locked()
    finally:
        lock.release()


def test_release_removes_the_file(tmp_path):
    lock = PidLock(tmp_path / "gustav.pid")
    lock.acquire()
    lock.release()
    assert not (tmp_path / "gustav.pid").exists()
    assert not lock.is_locked()


def test_second_daemon_in_another_process_is_refused(tmp_path, python_exe):
    """The real case: a live daemon elsewhere must block a second start."""
    path = tmp_path / "gustav.pid"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str((__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))!r})
        from gustav.core.lockfile import PidLock
        lock = PidLock({str(path)!r})
        lock.acquire()
        print("holding", flush=True)
        time.sleep(30)
        """
    )
    proc = subprocess.Popen(
        [python_exe, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout.readline().strip() == "holding"
        with pytest.raises(LockHeld) as exc:
            PidLock(path).acquire()
        assert "already running" in str(exc.value)
        assert exc.value.info["pid"] == proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_lock_from_a_dead_process_is_reclaimed(tmp_path, python_exe):
    """kill -9 must not leave a lock that needs manual cleanup."""
    path = tmp_path / "gustav.pid"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str((__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))!r})
        from gustav.core.lockfile import PidLock
        PidLock({str(path)!r}).acquire()
        print("holding", flush=True)
        time.sleep(30)
        """
    )
    proc = subprocess.Popen([python_exe, "-c", script], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "holding"
    proc.kill()
    proc.wait(timeout=5)

    # The stale file is still there, but the lock is not held.
    assert path.exists()
    lock = PidLock(path)
    assert not lock.is_locked()
    lock.acquire()
    assert lock.read()["pid"] == os.getpid()
    lock.release()


def test_reading_a_garbage_lock_file_does_not_raise(tmp_path):
    path = tmp_path / "gustav.pid"
    path.write_text("not json at all", encoding="utf-8")
    assert PidLock(path).read() is None


def test_missing_lock_file_reads_as_none(tmp_path):
    assert PidLock(tmp_path / "nope.pid").read() is None
