"""Stage 2 through the running daemon: detection, planning, approvals, thermal guard."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.core.scheduler import Observation
from gustav.core.state import State
from gustav.daemon import Daemon
from gustav.hw.detect import simulated
from gustav.hw.service import META_ACTIVE_BASE

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def daemon(project: Path) -> AsyncIterator[Daemon]:
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("small-12gb"))
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
async def test_the_daemon_plans_at_startup(daemon: Daemon):
    assert daemon.plan_outcome is not None
    plan = daemon.hw.stored_plan()
    assert plan is not None and plan.base_model == "qwen3-4b"
    assert daemon.db.get_meta(META_ACTIVE_BASE) == "qwen3-4b"


async def test_status_shows_the_plan_and_the_budget(client):
    hardware = (await client.get("/status")).json()["hardware"]
    assert "qwen3-4b" in hardware["plan"]
    assert hardware["base_model"] == "qwen3-4b"
    assert hardware["vram_inference_mb"] > 0
    assert hardware["expansion_budget"]["max_adapters"] >= 1
    assert hardware["fingerprint"]
    assert any("Simulated" in g or "GPU0" in g for g in hardware["gpus"])


async def test_planning_failure_does_not_stop_the_daemon(project):
    """A machine the planner cannot serve must still boot and answer."""
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("cpu-only"))
    await instance.start()
    try:
        assert instance.state.state is State.WAKE
        assert instance.plan_outcome.error
        transport = httpx.AsyncHTTPTransport(uds=str(instance.socket_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
            assert (await http.post("/chat", json={"message": "hi"})).json()["reply"]
            status = (await http.get("/status")).json()
            assert status["hardware"]["problem"]
    finally:
        await instance.shutdown("done")


# --- the hardware endpoints -----------------------------------------------
async def test_hardware_endpoint_describes_the_machine(client):
    info = (await client.get("/hardware")).json()
    assert info["source"] == "simulated"
    assert info["budgets"] and "usable of" in info["budgets"][0]
    assert info["host"]["ram_total_mb"] > 0
    assert info["plan"]["base_model"] == "qwen3-4b"


async def test_plan_endpoint_is_a_dry_run_by_default(daemon: Daemon, client):
    before = daemon.hw.stored_plan().base_model
    body = (await client.post("/hardware/plan", json={"balance": 0.0})).json()
    assert not body["applied"]
    assert body["plan"]["base_model"] != before
    assert daemon.hw.stored_plan().base_model == before, "a dry run must change nothing"
    assert body["explanation"] and body["rejected"]


async def test_plan_endpoint_can_apply(daemon: Daemon, client):
    body = (await client.post("/hardware/plan", json={"balance": 0.0, "apply": True})).json()
    assert body["applied"]
    assert daemon.hw.stored_plan().base_model == body["plan"]["base_model"]


async def test_plan_endpoint_accepts_overrides(client):
    body = (
        await client.post(
            "/hardware/plan",
            json={"overrides": {"prefer_model": "qwen3-1.7b"}},
        )
    ).json()
    assert body["plan"]["base_model"] == "qwen3-1.7b"


async def test_plan_endpoint_reports_an_impossible_request(client):
    response = await client.post(
        "/hardware/plan", json={"overrides": {"min_tokens_per_second": 99999}}
    )
    assert response.status_code == 409
    assert "no model on the ladder fits" in response.json()["detail"]


async def test_plan_endpoint_rejects_a_bad_override(client):
    response = await client.post("/hardware/plan", json={"overrides": {"balance": 7}})
    assert response.status_code == 400


# --- base-model approvals -------------------------------------------------
async def test_a_hardware_upgrade_waits_for_approval(project):
    """Boot on a small card, then boot the same data dir on a server."""
    config, paths = load_config(project / "config.yaml")
    small = Daemon(config, paths, detector=simulated("tiny-4gb"))
    await small.start()
    small_base = small.hw.active_base_model
    await small.shutdown("moving to the server")

    config, paths = load_config(project / "config.yaml")
    big = Daemon(config, paths, detector=simulated("server-multigpu"))
    await big.start()
    try:
        assert big.hw.active_base_model == small_base, "the base must not change on its own"
        approvals = big.db.query("SELECT * FROM approvals WHERE status = 'pending'")
        assert len(approvals) == 1
        assert approvals[0]["kind"] == "base_change"

        transport = httpx.AsyncHTTPTransport(uds=str(big.socket_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
            listed = (await http.get("/approvals")).json()["approvals"]
            assert listed[0]["subject"].startswith(small_base)
            assert (await http.get("/status")).json()["pending_approvals"]

            decided = (
                await http.post(f"/approvals/{approvals[0]['id']}", json={"approve": True})
            ).json()
            assert decided["approved"]
            assert big.hw.active_base_model != small_base
    finally:
        await big.shutdown("done")


async def test_an_approval_can_be_rejected(project):
    config, paths = load_config(project / "config.yaml")
    small = Daemon(config, paths, detector=simulated("tiny-4gb"))
    await small.start()
    await small.shutdown("moving")

    config, paths = load_config(project / "config.yaml")
    big = Daemon(config, paths, detector=simulated("server-multigpu"))
    await big.start()
    try:
        approval_id = big.db.query_one("SELECT id FROM approvals")["id"]
        base_before = big.hw.active_base_model
        transport = httpx.AsyncHTTPTransport(uds=str(big.socket_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
            body = (
                await http.post(f"/approvals/{approval_id}", json={"approve": False,
                                                                   "note": "later"})
            ).json()
            assert body["rejected"]
        assert big.hw.active_base_model == base_before
    finally:
        await big.shutdown("done")


async def test_an_unknown_approval_is_a_404(client):
    assert (await client.post("/approvals/999", json={"approve": True})).status_code == 404


# --- the thermal guard ----------------------------------------------------
async def test_a_hot_gpu_defers_sleep(daemon: Daemon, monkeypatch):
    from gustav.config.schema import ScheduleConfig
    from gustav.core.scheduler import Scheduler

    monkeypatch.setattr(daemon.hw, "hottest_gpu_c", lambda: 95.0)
    observation = daemon._observe()
    assert observation.gpu_temp_c == 95.0, "the daemon feeds the real temperature in"

    # The fixture's config has sleep switched off, so check the guard directly.
    scheduler = Scheduler(ScheduleConfig(enabled=True, min_idle_seconds=0,
                                         cooldown_minutes=0, max_gpu_temp_c=83))
    decision = scheduler.decide(observation)
    assert not decision.should_sleep and "95C" in decision.reason

    monkeypatch.setattr(daemon.hw, "hottest_gpu_c", lambda: 55.0)
    assert "95C" not in scheduler.decide(daemon._observe()).reason


async def test_a_broken_temperature_probe_is_not_fatal(daemon: Daemon, monkeypatch):
    def boom():
        raise OSError("nvml went away")

    monkeypatch.setattr(daemon.hw, "hottest_gpu_c", boom)
    assert daemon._observe().gpu_temp_c is None


# --- a pinned profile -----------------------------------------------------
async def test_a_pinned_profile_is_used_instead_of_planning(project, repo_root):
    (project / "profiles").mkdir(exist_ok=True)
    (project / "profiles" / "pinned.yaml").write_text(
        (repo_root / "profiles" / "tiny-4gb.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / "config.yaml").write_text(
        "logging:\n  console: false\n"
        "schedule:\n  enabled: false\n"
        "hardware:\n  profile: pinned\n",
        encoding="utf-8",
    )
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("small-12gb"))
    await instance.start()
    try:
        plan = instance.hw.stored_plan()
        assert plan.base_model == "qwen3-1.7b", "the profile wins over the planner"
        assert plan.kv_cache_dtype == "int8"
        assert instance.plan_outcome.source == "config"
    finally:
        await instance.shutdown("done")
