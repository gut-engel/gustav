"""Crash-recovery tests that run the real ``python daemon.py`` as a separate process.

These are the tests that matter most for an always-on daemon: ``kill -9`` at any
point must not lose or corrupt anything, and the next start must pick up where it
stopped.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

from gustav.core.journal import Journal, RunStatus, StepStatus
from gustav.sleep.pipeline import default_pipeline
from gustav.config.schema import Config
from gustav.store.db import Database

pytestmark = pytest.mark.slow

STEP_NAMES = [s.spec.name for s in default_pipeline(Config()).steps]
START_TIMEOUT = 30.0


class DaemonProcess:
    """Starts and stops the daemon the way a user would."""

    def __init__(self, repo_root: Path, project: Path, python_exe: str, **env: str):
        self.repo_root = repo_root
        self.project = project
        self.python_exe = python_exe
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", **env}
        self.env.pop("GUSTAV_FAULT", None)
        self.env.update({k: v for k, v in env.items()})
        self.proc: subprocess.Popen | None = None
        self.log = project / "var" / "daemon.out"

    @property
    def socket_path(self) -> Path:
        return self.project / "var" / "gustav.sock"

    def start(self, *, wait: bool = True) -> subprocess.Popen:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log.open("ab")
        self.proc = subprocess.Popen(
            [self.python_exe, str(self.repo_root / "daemon.py"), "--no-console-log"],
            cwd=self.project,
            env=self.env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        if wait:
            self.wait_until_listening()
        return self.proc

    def wait_until_listening(self, timeout: float = START_TIMEOUT) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"daemon exited with {self.proc.returncode}:\n{self._log_tail()}"
                )
            if self.socket_path.exists() and self._connects():
                return
            time.sleep(0.05)
        raise AssertionError(f"daemon never started listening:\n{self._log_tail()}")

    def _connects(self) -> bool:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.3)
                sock.connect(str(self.socket_path))
            return True
        except OSError:
            return False

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        import httpx

        transport = httpx.HTTPTransport(uds=str(self.socket_path))
        with httpx.Client(transport=transport, base_url="http://gustav", timeout=30) as client:
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

    def kill9(self) -> None:
        assert self.proc is not None
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait(timeout=10)

    def terminate(self, timeout: float = 30.0) -> int:
        assert self.proc is not None
        self.proc.send_signal(signal.SIGTERM)
        return self.proc.wait(timeout=timeout)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.terminate(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def _log_tail(self, lines: int = 40) -> str:
        if not self.log.exists():
            return "(no log)"
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-lines:])


@pytest.fixture
def daemon_process(repo_root, project, python_exe):
    created: list[DaemonProcess] = []

    def build(**env: str) -> DaemonProcess:
        proc = DaemonProcess(repo_root, project, python_exe, **env)
        created.append(proc)
        return proc

    yield build
    for proc in created:
        proc.stop()


def journal_for(project: Path) -> tuple[Database, Journal]:
    db = Database(project / "var" / "gustav.db")
    return db, Journal(db)


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


# ---------------------------------------------------------------------------


def test_daemon_starts_and_answers_over_the_socket(daemon_process):
    daemon = daemon_process()
    daemon.start()
    assert daemon.request("GET", "/health")["ok"]
    assert daemon.request("POST", "/chat", {"message": "hi"})["reply"].endswith("hi")


def test_a_second_daemon_refuses_to_start(daemon_process, project):
    first = daemon_process()
    first.start()
    second = daemon_process()
    proc = second.start(wait=False)
    assert proc.wait(timeout=30) != 0
    log = second._log_tail()
    assert "already running" in log


def test_sigterm_shuts_down_cleanly(daemon_process, project):
    daemon = daemon_process()
    daemon.start()
    daemon.request("POST", "/chat", {"message": "hello"})
    assert daemon.terminate() == 0
    assert not (project / "var" / "gustav.sock").exists()
    assert not (project / "var" / "gustav.pid").exists()

    db, journal = journal_for(project)
    history = db.query("SELECT to_state FROM state_history ORDER BY id")
    assert [r["to_state"] for r in history][-1] == "stopped"
    db.close()


def test_sigterm_during_sleep_pauses_and_keeps_progress(daemon_process, project):
    daemon = daemon_process(GUSTAV_FAULT="training=pause")
    daemon.start()
    run_id = daemon.request("POST", "/control/sleep", {})["run_id"]
    db, journal = journal_for(project)
    wait_for(lambda: journal.load_run(run_id).step("training").status is StepStatus.RUNNING)
    assert daemon.terminate() == 0

    run = journal.load_run(run_id)
    assert run.status is RunStatus.PAUSED
    assert run.step("snapshot").status is StepStatus.DONE
    db.close()


def test_kill9_leaves_no_stale_lock(daemon_process, project):
    daemon = daemon_process()
    daemon.start()
    pid_file = project / "var" / "gustav.pid"
    assert json.loads(pid_file.read_text())["pid"] == daemon.proc.pid
    daemon.kill9()
    assert pid_file.exists()  # the file is still there...

    again = daemon_process()
    again.start()  # ...but the lock is not held, so this works
    assert again.request("GET", "/health")["ok"]


@pytest.mark.parametrize("victim", STEP_NAMES)
def test_kill9_at_every_step_resumes_correctly(daemon_process, project, victim):
    """kill -9 while each step runs, then restart and finish the night."""
    first = daemon_process(GUSTAV_FAULT=f"{victim}=pause")
    first.start()
    run_id = first.request("POST", "/control/sleep", {})["run_id"]

    db, journal = journal_for(project)
    wait_for(
        lambda: journal.load_run(run_id).step(victim).status is StepStatus.RUNNING,
        timeout=60,
    )
    before = journal.load_run(run_id)
    done_before = {s.name: s.attempts for s in before.steps if s.status is StepStatus.DONE}
    first.kill9()

    # Restart without the fault: recovery decides what to do with the dead step.
    second = daemon_process()
    second.start()

    after_recovery = journal.load_run(run_id)
    assert after_recovery.status in (RunStatus.INTERRUPTED, RunStatus.FAILED)
    # Steps that had finished stay finished.
    for name, attempts in done_before.items():
        step = after_recovery.step(name)
        assert step.status is StepStatus.DONE, f"{name} lost its completion"
        assert step.attempts == attempts, f"{name} was re-run"

    policy = {s.spec.name: s.spec.on_crash for s in default_pipeline(Config()).steps}[victim]
    status = second.request("GET", "/status")

    if policy == "rollback":
        # A weight-changing step that died is rolled back and the night is abandoned.
        assert after_recovery.status is RunStatus.FAILED
        assert after_recovery.step(victim).status is StepStatus.FAILED
        assert run_id not in status["sleep"]["resumable_runs"]
        db.close()
        return

    assert run_id in status["sleep"]["resumable_runs"]
    resumed = second.request("POST", "/control/sleep", {})
    assert resumed["run_id"] == run_id, "resuming must continue the same run"
    wait_for(
        lambda: journal.load_run(run_id).status is RunStatus.COMPLETED,
        timeout=120,
    )
    final = journal.load_run(run_id)
    assert all(
        s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in final.steps
    ), [(s.name, str(s.status)) for s in final.steps]
    db.close()


def test_kill9_mid_step_keeps_the_checkpoint_for_resume_policies(daemon_process, project):
    daemon = daemon_process(GUSTAV_FAULT="training=pause")
    daemon.start()
    run_id = daemon.request("POST", "/control/sleep", {})["run_id"]
    db, journal = journal_for(project)
    wait_for(
        lambda: (journal.load_run(run_id).step("training").checkpoint or {}).get("unit") == 2,
        timeout=60,
    )
    daemon.kill9()

    again = daemon_process()
    again.start()
    step = journal.load_run(run_id).step("training")
    assert step.status is StepStatus.PENDING
    assert step.checkpoint == {"unit": 2, "step": "training"}

    again.request("POST", "/control/sleep", {})
    wait_for(lambda: journal.load_run(run_id).status is RunStatus.COMPLETED, timeout=120)
    assert journal.load_run(run_id).step("training").result["resumed_from"] == 2
    db.close()


def test_half_written_staging_is_cleaned_on_restart(daemon_process, project):
    staging = project / "var" / "tmp" / "adapter-coding-v004.partial"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "adapter_model.safetensors.part").write_bytes(b"\x00" * 128)

    daemon = daemon_process()
    daemon.start()
    status = daemon.request("GET", "/status")
    assert "staging" in status["recovery"]["summary"]
    assert not staging.exists()
