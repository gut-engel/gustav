"""The ``ai`` CLI, exercised against a real running daemon."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gustav.cli import build_parser, main

from .test_daemon_crash import DaemonProcess  # noqa: TID252 - shared process helper

pytestmark = pytest.mark.slow


@pytest.fixture
def running(repo_root, project, python_exe):
    daemon = DaemonProcess(repo_root, project, python_exe)
    daemon.start()
    try:
        yield daemon
    finally:
        daemon.stop()


def ai(project: Path, *args: str, repo_root: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    env.pop("GUSTAV_FAULT", None)
    return subprocess.run(
        [sys.executable, "-m", "gustav.cli", *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_status_renders_for_humans(running, project, repo_root):
    result = ai(project, "status", repo_root=repo_root)
    assert result.returncode == 0, result.stderr
    assert "state            wake" in result.stdout
    assert "sleep windows    00:00-06:00" in result.stdout
    assert "last backup      none yet" in result.stdout


def test_status_json_is_machine_readable(running, project, repo_root):
    import json

    result = ai(project, "status", "--json", repo_root=repo_root)
    assert json.loads(result.stdout)["state"] == "wake"


def test_chat_round_trip(running, project, repo_root):
    result = ai(project, "chat", "hello", "world", repo_root=repo_root)
    assert result.returncode == 0
    assert "hello world" in result.stdout


def test_sleep_then_inspect_the_run(running, project, repo_root):
    result = ai(project, "sleep", "now", repo_root=repo_root)
    assert result.returncode == 0, result.stderr
    assert "sleeping now (run 1)" in result.stdout

    listing = ai(project, "runs", repo_root=repo_root)
    assert "   1  " in listing.stdout

    detail = ai(project, "runs", "1", repo_root=repo_root)
    assert "snapshot" in detail.stdout


def test_pause_and_resume_learning(running, project, repo_root):
    assert "learning paused" in ai(project, "pause-learning", repo_root=repo_root).stdout
    refused = ai(project, "sleep", "now", repo_root=repo_root)
    assert refused.returncode == 1
    assert "paused" in refused.stderr
    assert "learning resumed" in ai(project, "resume-learning", repo_root=repo_root).stdout


def test_events_are_printed(running, project, repo_root):
    result = ai(project, "events", "--kind", "daemon.start", repo_root=repo_root)
    assert "daemon.start" in result.stdout


def test_stop_shuts_the_daemon_down(running, project, repo_root):
    result = ai(project, "stop", repo_root=repo_root)
    assert result.returncode == 0
    assert running.proc.wait(timeout=30) == 0


def test_a_missing_daemon_says_how_to_start_one(project, repo_root):
    result = ai(project, "status", repo_root=repo_root)
    assert result.returncode == 3
    assert "ai start" in result.stderr


def test_later_stage_commands_explain_themselves(project, repo_root):
    result = ai(project, "plan-hardware", repo_root=repo_root)
    assert result.returncode == 2
    assert "stage 2" in result.stderr


def test_parser_covers_the_documented_command_surface():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    names = set(actions[0].choices)
    for expected in (
        "chat", "status", "sleep", "wake", "pause-learning", "resume-learning",
        "report", "backup", "approve-expansion", "approve-base-change",
        "plan-hardware", "optimize", "expansion", "facts", "start", "stop",
    ):
        assert expected in names, expected
