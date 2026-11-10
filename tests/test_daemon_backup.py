"""Backups through the running daemon, including crash recovery of a promotion."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.backup.atomic import atomic_symlink
from gustav.config.load import load_config
from gustav.daemon import Daemon
from gustav.hw.detect import simulated
from gustav.testing.fakes import write_fake_adapter

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def daemon(project: Path) -> AsyncIterator[Daemon]:
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("small-12gb"))
    write_fake_adapter(instance.paths.adapters, "coding", "v001", base_model="qwen3-4b")
    write_fake_adapter(instance.paths.adapters, "coding", "v002", base_model="qwen3-4b")
    atomic_symlink(instance.paths.adapters / "coding" / "active", "v002")
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown("test finished")


@pytest_asyncio.fixture
async def adapter_daemon(project: Path) -> AsyncIterator[Daemon]:
    """A daemon whose responder is the real harness over a fake backend.

    The echo responder has no adapters, so this is what exercises the guarded
    activate/rollback path — all of it real except the weights themselves.
    """
    from gustav.inference.harness import Harness
    from gustav.testing.fakes import FakeBackend

    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("small-12gb"))
    instance.responder = Harness(
        config=config,
        paths=paths,
        hardware=instance.hw,
        events=instance.events,
        telemetry=instance.telemetry,
        db=instance.db,
        backend=FakeBackend(base_model="qwen3-4b"),
    )
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    write_fake_adapter(paths.adapters, "coding", "v002", base_model="qwen3-4b")
    atomic_symlink(paths.adapters / "coding" / "active", "v002")
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


# --- the API --------------------------------------------------------------
async def test_a_fresh_install_has_no_snapshots(client):
    payload = (await client.get("/backups")).json()
    assert payload["snapshots"] == []
    assert "keep the last" in payload["policy"]
    assert payload["free_mb"] > 0


async def test_creating_and_verifying_a_snapshot(client):
    created = (await client.post("/backups", json={"operation": "manual"})).json()
    assert created["verified"] and created["file_count"] > 0

    listed = (await client.get("/backups")).json()["snapshots"]
    assert [s["id"] for s in listed] == [created["id"]]

    verified = (await client.post(f"/backups/{created['id']}/verify")).json()
    assert verified["ok"] and not verified["problems"]


async def test_a_dry_run_reports_without_copying(client):
    result = (
        await client.post("/backups", json={"operation": "manual", "dry_run": True})
    ).json()
    assert result["dry_run"] and result["sources"]
    assert (await client.get("/backups")).json()["snapshots"] == []


async def test_verifying_an_unknown_snapshot_is_a_404(client):
    assert (await client.post("/backups/nope/verify")).status_code == 404


async def test_status_shows_the_last_backup(daemon, client):
    await client.post("/backups", json={"operation": "manual"})
    status = (await client.get("/status")).json()
    assert status["last_backup"]["operation"] == "manual"
    assert status["last_backup"]["verified"]


# --- the guard, through the daemon ----------------------------------------
async def test_activating_an_adapter_takes_a_snapshot_first(adapter_daemon):
    """Section 9: promotion is a weight change, so it is backed up first."""
    daemon = adapter_daemon
    result = await daemon.activate_adapter("coding", "v001")
    assert result["snapshot"], "no snapshot was taken for a weight change"
    assert os.readlink(daemon.paths.adapters / "coding" / "active") == "v001"

    row = daemon.db.query_one("SELECT * FROM weight_changes ORDER BY id DESC LIMIT 1")
    assert row["status"] == "done" and row["snapshot_id"] == result["snapshot"]
    assert daemon.backups.verify(result["snapshot"]).ok


async def test_rolling_back_is_also_guarded(adapter_daemon):
    result = await adapter_daemon.rollback_adapter("coding")
    assert result["version"] == "v001" and result["snapshot"]


async def test_a_failed_promotion_is_rolled_back_by_the_guard(adapter_daemon, monkeypatch):
    """If the hot-swap fails, the symlink must go back to where it was."""
    daemon = adapter_daemon
    before = os.readlink(daemon.paths.adapters / "coding" / "active")

    async def explode(module, path, *, version):
        raise RuntimeError("the adapter would not attach")

    monkeypatch.setattr(daemon.responder.backend, "load_adapter", explode)
    with pytest.raises(RuntimeError, match="would not attach"):
        await daemon.activate_adapter("coding", "v001")

    assert os.readlink(daemon.paths.adapters / "coding" / "active") == before
    row = daemon.db.query_one("SELECT status FROM weight_changes ORDER BY id DESC LIMIT 1")
    assert row["status"] == "rolled_back"


async def test_a_weight_change_is_refused_when_the_backup_cannot_be_taken(adapter_daemon):
    """No verified backup, no change — the body never runs."""
    daemon = adapter_daemon
    before = os.readlink(daemon.paths.adapters / "coding" / "active")
    daemon.backups.min_free_mb = 10**9  # as if the disk were full

    from gustav.backup.guard import BackupRequired

    with pytest.raises(BackupRequired, match="Nothing was changed"):
        await daemon.activate_adapter("coding", "v001")
    assert os.readlink(daemon.paths.adapters / "coding" / "active") == before


async def test_a_snapshot_referenced_by_a_pending_approval_is_protected(daemon, client):
    created = (await client.post("/backups", json={"operation": "manual"})).json()
    daemon.db.insert(
        "approvals",
        {"kind": "base_change", "subject": "a -> b", "snapshot_id": created["id"]},
    )
    listed = (await client.get("/backups")).json()["snapshots"]
    assert listed[0]["protected"]
    assert created["id"] in daemon.protected_snapshots()


# --- restore --------------------------------------------------------------
async def test_restore_brings_an_adapter_back(daemon, client):
    import shutil

    created = (await client.post("/backups", json={"operation": "before"})).json()
    shutil.rmtree(daemon.paths.adapters / "coding")

    result = (
        await client.post(f"/backups/{created['id']}/restore", json={"only": ["adapters"]})
    ).json()
    assert "adapters" in result["restored"]
    assert result["undo_snapshot"], "a restore must itself be undoable"
    assert (daemon.paths.adapters / "coding" / "v002").exists()


async def test_restore_releases_the_gpu_first(daemon, client):
    """Files must not be swapped underneath a loaded model."""
    created = (await client.post("/backups", json={"operation": "before"})).json()
    await daemon._ensure_loaded()
    assert daemon.responder.loaded
    await client.post(f"/backups/{created['id']}/restore", json={"only": ["adapters"]})
    assert not daemon.responder.loaded


async def test_a_dry_run_restore_changes_nothing(daemon, client):
    created = (await client.post("/backups", json={"operation": "before"})).json()
    result = (
        await client.post(
            f"/backups/{created['id']}/restore", json={"dry_run": True}
        )
    ).json()
    assert result["dry_run"] and result["undo_snapshot"] is None
    assert os.readlink(daemon.paths.adapters / "coding" / "active") == "v002"


# --- retention ------------------------------------------------------------
async def test_retention_keeps_the_newest_and_the_policy_window(daemon, client):
    ids = []
    for index in range(8):
        ids.append((await client.post("/backups", json={"operation": f"run{index}"})).json()["id"])

    result = (await client.post("/backups/retention", params={"dry_run": True})).json()
    assert ids[-1] in result["kept"], "the newest snapshot is never deleted"
    assert len(result["kept"]) >= daemon.config.backup.keep_last


async def test_retention_actually_deletes_when_asked(daemon, client):
    for index in range(8):
        await client.post("/backups", json={"operation": f"run{index}"})
    before = len((await client.get("/backups")).json()["snapshots"])
    result = (await client.post("/backups/retention")).json()
    after = len((await client.get("/backups")).json()["snapshots"])
    assert after == before - len(result["deleted"])


async def test_a_low_backup_location_warns_instead_of_deleting(project):
    """Losing backups to free space is the failure backups exist to prevent."""
    (project / "config.yaml").write_text(
        "logging:\n  console: false\nschedule:\n  enabled: false\n"
        "inference:\n  responder: echo\n"
        "backup:\n  warn_free_mb: 100000000\n",
        encoding="utf-8",
    )
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("small-12gb"))
    await instance.start()
    try:
        instance.apply_retention(dry_run=True)
        assert any(e["kind"] == "backup.low_space" for e in instance.events.recent())
    finally:
        await instance.shutdown("done")


# --- crash recovery of a promotion ----------------------------------------
async def test_an_interrupted_promotion_is_undone_at_startup(project):
    """The dangerous case: killed between the snapshot and the finished promotion."""
    config, paths = load_config(project / "config.yaml")
    first = Daemon(config, paths, detector=simulated("small-12gb"))
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    atomic_symlink(paths.adapters / "coding" / "active", "v001")
    await first.start()
    snapshot = first.backups.create("promote coding v002")
    first.db.insert(
        "weight_changes",
        {"operation": "promote coding v002", "snapshot_id": snapshot.id,
         "sources": "adapters", "status": "in_progress"},
    )
    # ...and the crash left a half-made version live.
    write_fake_adapter(paths.adapters, "coding", "v002", base_model="qwen3-4b")
    atomic_symlink(paths.adapters / "coding" / "active", "v002")
    await first.shutdown("kill -9 stand-in")

    config, paths = load_config(project / "config.yaml")
    second = Daemon(config, paths, detector=simulated("small-12gb"))
    await second.start()
    try:
        assert os.readlink(paths.adapters / "coding" / "active") == "v001"
        assert not (paths.adapters / "coding" / "v002").exists()
        row = second.db.query_one(
            "SELECT status FROM weight_changes ORDER BY id DESC LIMIT 1"
        )
        assert row["status"] == "rolled_back"
        assert any(e["kind"] == "weights.recovered" for e in second.events.recent())
    finally:
        await second.shutdown("done")


async def test_a_clean_startup_recovers_nothing(daemon):
    assert not any(
        e["kind"] in ("weights.recovered", "weights.recovery_failed")
        for e in daemon.events.recent()
    )


async def test_a_base_model_change_is_backed_up_before_it_is_applied(project):
    """Section 9 names this explicitly: it invalidates every adapter."""
    import json

    from gustav.hw.service import META_ACTIVE_BASE

    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("tiny-4gb"))
    await instance.start()
    try:
        # Pretend the planner proposed a bigger base model.
        plan = instance.hw.stored_plan()
        proposed = {**plan.to_dict(), "base_model": "qwen3-8b"}
        approval_id = instance.db.insert(
            "approvals",
            {
                "kind": "base_change",
                "subject": f"{instance.hw.active_base_model} -> qwen3-8b",
                "payload": json.dumps(proposed),
            },
        )

        result = instance.approve_base_change(approval_id)
        assert result["approved"]
        assert result["snapshot"], "a base change must take a snapshot first"
        assert instance.backups.verify(result["snapshot"]).ok
        assert instance.db.get_meta(META_ACTIVE_BASE) == "qwen3-8b"

        # The snapshot is recorded on the approval, and therefore protected.
        row = instance.db.query_one(
            "SELECT snapshot_id, status FROM approvals WHERE id = ?", (approval_id,)
        )
        assert row["snapshot_id"] == result["snapshot"] and row["status"] == "approved"
        assert any(
            e["kind"] == "hardware.base_change_approved" for e in instance.events.recent()
        )
    finally:
        await instance.shutdown("done")
