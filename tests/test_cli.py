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
    # Simulated hardware so the CLI's output does not depend on this machine's GPU.
    daemon = DaemonProcess(
        repo_root, project, python_exe, GUSTAV_SIMULATE_HARDWARE="small-12gb"
    )
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


def test_every_command_the_readme_documents_exists(repo_root):
    """The README's command table and the parser cannot drift apart."""
    import re

    from gustav.cli import NOT_YET

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    table = readme[readme.index("## Commands"):readme.index("## Configuration")]
    documented = {
        match.group(1)
        for match in re.finditer(r"`ai ([a-z][a-z-]*)", table)
    }
    parser = build_parser()
    known = set(next(a for a in parser._actions if a.dest == "command").choices)
    assert documented, "the command table should not be empty"
    assert documented <= known, f"documented but missing: {sorted(documented - known)}"
    assert not NOT_YET, f"these commands still say 'not yet': {sorted(NOT_YET)}"


def test_facts_and_skills_are_honest_when_nothing_was_learned_yet(running, project, repo_root):
    facts = ai(project, "facts", repo_root=repo_root)
    assert facts.returncode == 0, facts.stderr
    assert "not learned any facts yet" in facts.stdout
    skills = ai(project, "skills", repo_root=repo_root)
    assert "nothing has been measured yet" in skills.stdout


def test_a_report_can_be_written_for_a_run(running, project, repo_root):
    assert ai(project, "sleep", "now", repo_root=repo_root).returncode == 0
    result = ai(project, "report", repo_root=repo_root)
    assert result.returncode == 0, result.stderr
    assert "report for run 1" in result.stdout
    written = Path(result.stdout.split(": ", 1)[1].strip())
    assert written.exists() and "Sleep run 1" in written.read_text(encoding="utf-8")


def test_a_report_for_a_run_that_does_not_exist_says_so(running, project, repo_root):
    result = ai(project, "report", "99", repo_root=repo_root)
    assert result.returncode == 1
    assert "no sleep run 99" in result.stderr


def test_plan_hardware_explains_its_choice(running, project, repo_root):
    result = ai(project, "plan-hardware", "-v", repo_root=repo_root)
    assert result.returncode == 0, result.stderr
    assert "qwen3-4b @ 4bit-nf4" in result.stdout
    assert "largest model on the ladder" in result.stdout
    assert "not chosen:" in result.stdout
    assert "pass --apply to store this plan" in result.stdout


def test_plan_hardware_balance_changes_the_answer(running, project, repo_root):
    fast = ai(project, "plan-hardware", "--balance", "0.0", repo_root=repo_root).stdout
    smart = ai(project, "plan-hardware", "--balance", "1.0", repo_root=repo_root).stdout
    assert "qwen3-1.7b" in fast
    assert "qwen3-8b" in smart


def test_plan_hardware_json(running, project, repo_root):
    import json

    payload = json.loads(ai(project, "plan-hardware", "--json", repo_root=repo_root).stdout)
    assert payload["plan"]["base_model"] == "qwen3-4b"
    assert payload["applied"] is False


def test_hardware_command_shows_what_was_detected(running, project, repo_root):
    result = ai(project, "hardware", repo_root=repo_root)
    assert "fingerprint" in result.stdout
    assert "usable of" in result.stdout
    assert "active base      qwen3-4b" in result.stdout


def test_status_includes_the_plan(running, project, repo_root):
    result = ai(project, "status", repo_root=repo_root)
    assert "hardware plan    qwen3-4b" in result.stdout
    assert "adapters <= r" in result.stdout


def test_approvals_is_empty_on_a_fresh_install(running, project, repo_root):
    assert "no pending approvals" in ai(project, "approvals", repo_root=repo_root).stdout


def test_parser_covers_the_documented_command_surface():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    names = set(actions[0].choices)
    for expected in (
        "chat", "status", "sleep", "wake", "pause-learning", "resume-learning",
        "report", "backup", "approve-expansion", "approve-base-change",
        "plan-hardware", "optimize", "expansion", "facts", "skills", "start", "stop",
    ):
        assert expected in names, expected


# --- offline planning (no daemon) ----------------------------------------
def test_plan_hardware_offline_needs_no_daemon(project, repo_root):
    result = ai(project, "plan-hardware", "--simulate", "small-12gb", repo_root=repo_root)
    assert result.returncode == 0, result.stderr
    assert "qwen3-4b" in result.stdout
    assert "computed locally" in result.stdout


def test_plan_hardware_offline_for_a_machine_that_cannot_run_it(project, repo_root):
    result = ai(project, "plan-hardware", "--simulate", "cpu-only", repo_root=repo_root)
    assert result.returncode == 1
    assert "no GPU was detected" in result.stderr


def test_plan_hardware_offline_respects_no_training(project, repo_root):
    result = ai(
        project, "plan-hardware", "--simulate", "tiny-4gb", "--no-training",
        repo_root=repo_root,
    )
    assert "inference only" in result.stdout
    assert "serves only" in result.stdout


# --- backups --------------------------------------------------------------
def test_backup_list_is_empty_on_a_fresh_install(running, project, repo_root):
    result = ai(project, "backup", "list", repo_root=repo_root)
    assert result.returncode == 0, result.stderr
    assert "(none yet)" in result.stdout
    assert "retention: keep the last" in result.stdout


def test_backup_create_verify_and_restore(running, project, repo_root):
    created = ai(project, "backup", "create", "--operation", "manual", repo_root=repo_root)
    assert created.returncode == 0, created.stderr
    assert "verified" in created.stdout
    snapshot_id = created.stdout.split()[1].rstrip(":")

    listed = ai(project, "backup", "list", repo_root=repo_root)
    assert snapshot_id in listed.stdout

    verified = ai(project, "backup", "verify", snapshot_id, repo_root=repo_root)
    assert verified.returncode == 0
    assert "all hashes match" in verified.stdout

    restored = ai(project, "backup", "restore", snapshot_id, "--dry-run", repo_root=repo_root)
    assert "would restore" in restored.stdout


def test_backup_create_dry_run_reports_sizes(running, project, repo_root):
    result = ai(project, "backup", "create", "--dry-run", repo_root=repo_root)
    assert "would go to" in result.stdout
    assert "free at the destination" in result.stdout
    assert "(none yet)" in ai(project, "backup", "list", repo_root=repo_root).stdout


def test_backup_verify_detects_tampering(running, project, repo_root):
    created = ai(project, "backup", "create", repo_root=repo_root)
    snapshot_id = created.stdout.split()[1].rstrip(":")
    target = next((project / "backups" / snapshot_id).rglob("gustav.db"))
    target.write_bytes(b"corrupted")

    result = ai(project, "backup", "verify", snapshot_id, repo_root=repo_root)
    assert result.returncode == 1
    assert "problem" in result.stdout


def test_backup_retention_dry_run(running, project, repo_root):
    for index in range(3):
        ai(project, "backup", "create", "--operation", f"run{index}", repo_root=repo_root)
    result = ai(project, "backup", "retention", "--dry-run", repo_root=repo_root)
    assert "policy: keep the last" in result.stdout
    assert "keeping 3" in result.stdout


def test_status_shows_the_last_backup(running, project, repo_root):
    ai(project, "backup", "create", "--operation", "manual", repo_root=repo_root)
    result = ai(project, "status", repo_root=repo_root)
    assert "last backup      " in result.stdout
    assert "verified" in result.stdout


def test_release_gpu_is_honest_when_nothing_is_loaded(running, project, repo_root):
    result = ai(project, "release-gpu", repo_root=repo_root)
    assert result.returncode == 0
    assert "already free" in result.stdout or "released" in result.stdout
