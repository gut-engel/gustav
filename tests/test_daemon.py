"""In-process daemon tests: the real daemon, the real control plane, no GPU."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.core.journal import RunStatus, StepStatus
from gustav.core.lockfile import LockHeld
from gustav.core.state import State
from gustav.daemon import Daemon
from gustav.wake.responder import EchoResponder

pytestmark = pytest.mark.asyncio


async def until(predicate: Callable[[], bool], timeout: float = 15.0, interval: float = 0.05):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


@pytest_asyncio.fixture
async def daemon(project: Path) -> AsyncIterator[Daemon]:
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths)
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown("test finished")


@pytest_asyncio.fixture
async def client(daemon: Daemon) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.AsyncHTTPTransport(uds=str(daemon.socket_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
        yield http


# --- startup --------------------------------------------------------------
async def test_daemon_comes_up_awake_and_serving(daemon: Daemon, client):
    assert daemon.state.state is State.WAKE
    health = (await client.get("/health")).json()
    assert health["ok"] and health["state"] == "wake"


async def test_the_lock_file_names_the_socket(daemon: Daemon):
    info = daemon.lock.read()
    assert info["pid"] == daemon.status()["pid"]
    assert Path(info["socket"]) == daemon.socket_path


async def test_a_second_daemon_on_the_same_data_dir_is_refused(daemon: Daemon, project):
    config, paths = load_config(project / "config.yaml")
    second = Daemon(config, paths)
    with pytest.raises(LockHeld):
        await second.start()
    second.db.close()


async def test_status_has_everything_ai_status_prints(client):
    status = (await client.get("/status")).json()
    for key in (
        "state", "pid", "learning_paused", "model", "gpu", "sleep",
        "pending_items", "failed_steps", "pending_approvals", "recovery",
        "hardware", "last_backup",
    ):
        assert key in status, key
    for key in ("fingerprint", "gpus", "plan", "base_model", "expansion_budget"):
        assert key in status["hardware"], key
    assert status["sleep"]["windows"] == ["00:00-06:00"]
    assert status["sleep"]["next"] in ("sleep disabled", "window open now") or status["sleep"][
        "next"
    ].startswith("20")


async def test_safe_mode_when_the_model_will_not_load(project):
    class BrokenResponder(EchoResponder):
        def health(self):
            return False, "adapter personal-memory v004 fails its health prompt"

    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, responder=BrokenResponder())
    await instance.start()
    try:
        assert instance.state.state is State.SAFE_MODE
        assert instance.recovery_report.safe_mode
        # Still serving: safe mode answers, it just will not learn.
        assert instance.state.serving
    finally:
        await instance.shutdown("done")


# --- wake phase -----------------------------------------------------------
async def test_chat_answers_and_logs_both_turns(daemon: Daemon, client):
    body = (await client.post("/chat", json={"message": "hello there"})).json()
    assert "hello there" in body["reply"]
    rows = daemon.db.query("SELECT role, content FROM turns ORDER BY id")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "hello there"


async def test_chat_records_pending_work_for_the_scheduler(daemon: Daemon, client):
    assert daemon.journal.pending_work_count() == 0
    await client.post("/chat", json={"message": "one"})
    await client.post("/chat", json={"message": "two"})
    assert daemon.journal.pending_work_count() == 2


async def test_chat_keeps_per_session_history(daemon: Daemon, client):
    await client.post("/chat", json={"message": "first", "session": "a"})
    body = (await client.post("/chat", json={"message": "second", "session": "a"})).json()
    other = (await client.post("/chat", json={"message": "hi", "session": "b"})).json()
    assert body["session"] != other["session"]
    assert daemon.db.scalar("SELECT COUNT(*) FROM sessions") == 2


async def test_empty_message_is_rejected(client):
    assert (await client.post("/chat", json={"message": ""})).status_code == 422


# --- sleep ----------------------------------------------------------------
async def test_manual_sleep_runs_the_whole_pipeline(daemon: Daemon, client):
    body = (await client.post("/control/sleep", json={})).json()
    assert body["accepted"]
    run_id = body["run_id"]

    await until(lambda: daemon.state.state is State.WAKE)
    run = daemon.journal.load_run(run_id)
    assert run.status is RunStatus.COMPLETED
    assert all(s.status is StepStatus.DONE for s in run.steps)
    assert daemon.db.get_meta("last_sleep_finished_at")


async def test_sleep_consumes_pending_work(daemon: Daemon, client):
    await client.post("/chat", json={"message": "something to learn from"})
    assert daemon.journal.pending_work_count() == 1
    await client.post("/control/sleep", json={})
    await until(lambda: daemon.state.state is State.WAKE)
    assert daemon.journal.pending_work_count() == 0


async def test_dry_run_leaves_no_artefacts(daemon: Daemon, client):
    body = (await client.post("/control/sleep", json={"dry_run": True})).json()
    await until(lambda: daemon.state.state is State.WAKE)
    run = daemon.journal.load_run(body["run_id"])
    assert run.dry_run
    assert all(s.result and s.result["dry_run"] for s in run.steps)


async def test_sleeping_twice_at_once_is_refused(daemon: Daemon, client, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "consolidation=pause")
    await client.post("/control/sleep", json={})
    await until(lambda: daemon.state.state is State.SLEEP)
    body = (await client.post("/control/sleep", json={})).json()
    assert not body["accepted"] and body["reason"] == "already asleep"
    await client.post("/control/wake")


async def test_a_user_message_pauses_sleep_and_gets_an_answer(
    daemon: Daemon, client, monkeypatch
):
    """The single-GPU case: training checkpoints, chat comes back, sleep resumes later."""
    monkeypatch.setenv("GUSTAV_FAULT", "training=pause")
    body = (await client.post("/control/sleep", json={})).json()
    run_id = body["run_id"]
    await until(
        lambda: daemon.journal.load_run(run_id).step("training").status is StepStatus.RUNNING
    )
    assert daemon.lease.holder == "sleep:training"

    reply = (await client.post("/chat", json={"message": "are you awake?"})).json()
    assert reply["sleep_paused"] is True
    assert "are you awake?" in reply["reply"]
    assert daemon.state.state is State.WAKE
    assert daemon.lease.holder == "inference"

    run = daemon.journal.load_run(run_id)
    assert run.status is RunStatus.PAUSED
    assert run.step("training").status is StepStatus.PREEMPTED
    assert run.step("training").checkpoint  # progress was not thrown away
    assert run.step("snapshot").status is StepStatus.DONE


async def test_a_paused_run_is_resumed_not_restarted(daemon: Daemon, client, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "training=pause")
    run_id = (await client.post("/control/sleep", json={})).json()["run_id"]
    await until(
        lambda: daemon.journal.load_run(run_id).step("training").status is StepStatus.RUNNING
    )
    await client.post("/chat", json={"message": "hi"})
    done_before = {s.name for s in daemon.journal.load_run(run_id).steps
                   if s.status is StepStatus.DONE}
    attempts_before = {s.name: s.attempts for s in daemon.journal.load_run(run_id).steps}

    monkeypatch.delenv("GUSTAV_FAULT")
    body = (await client.post("/control/sleep", json={})).json()
    assert body["run_id"] == run_id, "a resumed night must not start a new run"
    await until(lambda: daemon.journal.load_run(run_id).status is RunStatus.COMPLETED)

    after = daemon.journal.load_run(run_id)
    for name in done_before:
        assert after.step(name).attempts == attempts_before[name], f"{name} ran twice"


async def test_wake_pauses_sleep_on_demand(daemon: Daemon, client, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "study=pause")
    run_id = (await client.post("/control/sleep", json={})).json()["run_id"]
    await until(lambda: daemon.state.state is State.SLEEP)
    body = (await client.post("/control/wake")).json()
    assert body["accepted"]
    await until(lambda: daemon.state.state is State.WAKE)
    assert daemon.journal.load_run(run_id).status is RunStatus.PAUSED


async def test_wake_when_awake_is_a_no_op(client):
    body = (await client.post("/control/wake")).json()
    assert not body["accepted"] and "not asleep" in body["reason"]


async def test_pause_learning_blocks_sleep(daemon: Daemon, client):
    await client.post("/control/learning", json={"paused": True})
    body = (await client.post("/control/sleep", json={})).json()
    assert not body["accepted"] and "paused" in body["reason"]
    assert daemon.state.state is State.WAKE

    await client.post("/control/learning", json={"paused": False})
    assert (await client.post("/control/sleep", json={})).json()["accepted"]
    await until(lambda: daemon.state.state is State.WAKE)


async def test_a_failing_step_does_not_stop_the_night(daemon: Daemon, client, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "dreaming=fail")
    run_id = (await client.post("/control/sleep", json={})).json()["run_id"]
    await until(lambda: daemon.state.state is State.WAKE, timeout=30)
    run = daemon.journal.load_run(run_id)
    assert run.status is RunStatus.COMPLETED
    assert run.step("dreaming").status is StepStatus.FAILED
    assert run.step("training").status is StepStatus.DONE
    assert any(s["step"] == "dreaming" for s in daemon.status()["failed_steps"])


async def test_a_failing_critical_step_fails_the_run(daemon: Daemon, client, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "snapshot=fail")
    run_id = (await client.post("/control/sleep", json={})).json()["run_id"]
    await until(lambda: daemon.state.state is State.WAKE, timeout=30)
    run = daemon.journal.load_run(run_id)
    assert run.status is RunStatus.FAILED
    assert run.step("training").status is StepStatus.PENDING


async def test_a_crashing_child_leaves_chat_alive(daemon: Daemon, client, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "training=oom")
    await client.post("/control/sleep", json={})
    await until(lambda: daemon.state.state is State.WAKE, timeout=60)
    reply = (await client.post("/chat", json={"message": "still there?"})).json()
    assert "still there?" in reply["reply"]


# --- scheduler ------------------------------------------------------------
async def test_the_scheduler_triggers_sleep_by_itself(project, monkeypatch):
    """No cron: the daemon's own loop notices the pending-work threshold."""
    (project / "config.yaml").write_text(
        "logging:\n  console: false\n"
        "schedule:\n  enabled: true\n  windows: []\n  tick_seconds: 0.05\n"
        "  pending_threshold: 2\n  min_idle_seconds: 0\n  cooldown_minutes: 0\n"
        "  min_pending_items: 0\n"
        "supervisor:\n  memory_limit_mb: 0\n  max_attempts: 1\n"
        "sleep:\n  simulated_step_seconds: 0.01\n",
        encoding="utf-8",
    )
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths)
    await instance.start()
    try:
        transport = httpx.AsyncHTTPTransport(uds=str(instance.socket_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
            await http.post("/chat", json={"message": "one"})
            await http.post("/chat", json={"message": "two"})
            await until(lambda: instance.journal.latest_run() is not None, timeout=20)
            run = instance.journal.latest_run()
            assert run.trigger == "pending"
            await until(
                lambda: instance.journal.load_run(run.id).status is RunStatus.COMPLETED,
                timeout=30,
            )
    finally:
        await instance.shutdown("done")


# --- observability --------------------------------------------------------
async def test_events_are_readable_over_the_api(client):
    await client.post("/control/sleep", json={})
    payload = (await client.get("/events", params={"limit": 100})).json()
    kinds = {e["kind"] for e in payload["events"]}
    assert {"daemon.start", "sleep.started"} <= kinds


async def test_runs_endpoint_lists_steps(client):
    run_id = (await client.post("/control/sleep", json={})).json()["run_id"]
    listing = (await client.get("/runs")).json()["runs"]
    assert listing[0]["id"] == run_id
    detail = (await client.get(f"/runs/{run_id}")).json()
    assert [s["name"] for s in detail["steps"]][0] == "snapshot"
    assert detail["steps"][0]["on_crash"] == "restart"


async def test_unknown_run_is_a_404(client):
    assert (await client.get("/runs/999")).status_code == 404


async def test_state_history_is_recorded(daemon: Daemon, client):
    await client.post("/control/sleep", json={})
    await until(lambda: daemon.state.state is State.WAKE)
    rows = daemon.db.query("SELECT from_state, to_state FROM state_history ORDER BY id")
    seen = [(r["from_state"], r["to_state"]) for r in rows]
    assert ("starting", "wake") in seen
    assert ("wake", "sleep_pending") in seen
    assert ("sleep_pending", "sleep") in seen
    assert ("sleep", "wake") in seen


# --- shutdown -------------------------------------------------------------
async def test_graceful_shutdown_releases_everything(project):
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths)
    await instance.start()
    socket_path = instance.socket_path
    await instance.shutdown("graceful test")

    assert instance.state.state is State.STOPPED
    assert not socket_path.exists()
    assert not paths.lock_file.exists()
    assert not instance.responder.loaded  # the GPU was released


async def test_shutdown_during_sleep_checkpoints_instead_of_truncating(project, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "training=pause")
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths)
    await instance.start()
    transport = httpx.AsyncHTTPTransport(uds=str(instance.socket_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
        run_id = (await http.post("/control/sleep", json={})).json()["run_id"]
        await until(
            lambda: instance.journal.load_run(run_id).step("training").status
            is StepStatus.RUNNING
        )
    await instance.shutdown("sigterm")

    # Read the journal back from disk: whatever was written must have survived.
    from gustav.core.journal import Journal
    from gustav.store.db import open_db

    reopened = open_db(paths.db)
    run = Journal(reopened).load_run(run_id)
    assert run.status is RunStatus.PAUSED
    step = run.step("training")
    assert step.status in (StepStatus.PREEMPTED, StepStatus.PENDING)
    assert step.checkpoint, "a step stopped by shutdown must leave its progress behind"
    reopened.close()


async def test_shutdown_is_idempotent(daemon: Daemon):
    await daemon.shutdown("first")
    await daemon.shutdown("second")
    assert daemon.state.state is State.STOPPED
