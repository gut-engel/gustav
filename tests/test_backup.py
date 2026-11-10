"""Snapshots, verification, atomic promotion, restore, retention and the guard."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gustav.backup.atomic import atomic_symlink, atomic_write, promote, staged
from gustav.backup.guard import BackupRequired, WeightChangeGuard
from gustav.backup.manifest import Manifest, build_manifest, hash_file
from gustav.backup.restore import Restorer
from gustav.backup.retention import RetentionPolicy, plan_retention
from gustav.backup.snapshot import (
    BackupError,
    NotEnoughSpace,
    Snapshot,
    SnapshotStore,
)
from gustav.testing.fakes import write_fake_adapter


@pytest.fixture
def populated(paths, db):
    """A project with adapters, a registry, a database and a config."""
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    write_fake_adapter(paths.adapters, "coding", "v002", base_model="qwen3-4b")
    atomic_symlink(paths.adapters / "coding" / "active", "v002")
    write_fake_adapter(paths.adapters, "rust", "v001", base_model="qwen3-4b")
    atomic_symlink(paths.adapters / "rust" / "active", "v001")
    paths.registry.parent.mkdir(parents=True, exist_ok=True)
    paths.registry.write_text("modules:\n  - name: coding\n", encoding="utf-8")
    db.insert("events", {"kind": "test", "message": "before the snapshot"})
    return paths


@pytest.fixture
def store(populated, db, events):
    return SnapshotStore(
        populated, location=populated.root / "backups", db=db, events=events, min_free_mb=1
    )


# --- atomic primitives ----------------------------------------------------
def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = atomic_write(tmp_path / "a.txt", "hello")
    assert target.read_text() == "hello"
    assert not list(tmp_path.glob(".*.part"))


def test_atomic_symlink_never_disappears(tmp_path):
    (tmp_path / "v001").mkdir()
    (tmp_path / "v002").mkdir()
    link = tmp_path / "active"
    atomic_symlink(link, "v001")
    assert os.readlink(link) == "v001"
    atomic_symlink(link, "v002")
    assert os.readlink(link) == "v002"
    assert not list(tmp_path.glob(".*.swap"))


def test_promote_swaps_a_directory(tmp_path):
    staging = tmp_path / "new"
    staging.mkdir()
    (staging / "file").write_text("new", encoding="utf-8")
    target = tmp_path / "live"
    target.mkdir()
    (target / "file").write_text("old", encoding="utf-8")

    promote(staging, target)
    assert (target / "file").read_text() == "new"
    assert not staging.exists()
    assert not list(tmp_path.glob(".*.retired"))


def test_promote_into_an_empty_place_works(tmp_path):
    staging = tmp_path / "new"
    staging.mkdir()
    promote(staging, tmp_path / "live")
    assert (tmp_path / "live").is_dir()


def test_staged_cleans_up_after_a_failure(tmp_path):
    with pytest.raises(ValueError):
        with staged(tmp_path, "op") as staging:
            (staging / "half-written").write_text("x", encoding="utf-8")
            raise ValueError("boom")
    assert not list(tmp_path.glob("*.staging"))


# --- manifests ------------------------------------------------------------
def test_a_manifest_hashes_every_file(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta", encoding="utf-8")
    manifest = build_manifest(tmp_path, snapshot_id="s", operation="test")
    assert manifest.file_count == 2
    assert all(entry.sha256 for entry in manifest.files)
    assert manifest.verify(tmp_path) == []


def test_verification_catches_a_changed_file(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    manifest = build_manifest(tmp_path, snapshot_id="s", operation="test")
    (tmp_path / "a.txt").write_text("tampered", encoding="utf-8")
    problems = manifest.verify(tmp_path)
    assert [p.kind for p in problems] == ["size"]


def test_verification_catches_the_same_size_but_different_content(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    manifest = build_manifest(tmp_path, snapshot_id="s", operation="test")
    (tmp_path / "a.txt").write_text("ALPHA", encoding="utf-8")
    assert [p.kind for p in manifest.verify(tmp_path)] == ["hash"]


def test_verification_catches_a_missing_file(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    manifest = build_manifest(tmp_path, snapshot_id="s", operation="test")
    (tmp_path / "a.txt").unlink()
    assert [p.kind for p in manifest.verify(tmp_path)] == ["missing"]


def test_symlinks_are_recorded_as_symlinks_not_followed(tmp_path):
    (tmp_path / "v001").mkdir()
    (tmp_path / "v001" / "weights").write_text("w", encoding="utf-8")
    atomic_symlink(tmp_path / "active", "v001")
    manifest = build_manifest(tmp_path, snapshot_id="s", operation="test")
    link = next(f for f in manifest.files if f.path == "active")
    assert link.is_symlink and link.symlink_to == "v001"
    assert manifest.verify(tmp_path) == []


def test_a_repointed_symlink_fails_verification(tmp_path):
    (tmp_path / "v001").mkdir()
    (tmp_path / "v002").mkdir()
    atomic_symlink(tmp_path / "active", "v001")
    manifest = build_manifest(tmp_path, snapshot_id="s", operation="test")
    atomic_symlink(tmp_path / "active", "v002")
    assert [p.kind for p in manifest.verify(tmp_path)] == ["symlink"]


def test_a_manifest_round_trips(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    original = build_manifest(tmp_path, snapshot_id="s", operation="test")
    original.write(tmp_path)
    restored = Manifest.read(tmp_path)
    assert restored.id == "s" and restored.file_count == original.file_count


def test_hash_file_matches_sha256sum(tmp_path):
    import hashlib

    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00\x01\x02" * 1000)
    assert hash_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# --- creating snapshots ---------------------------------------------------
def test_a_snapshot_contains_what_section_nine_asks_for(store, populated):
    snapshot = store.create("test")
    names = {child.name for child in snapshot.path.iterdir() if child.is_dir()}
    assert {"adapters", "registry", "database"} <= names
    assert (snapshot.path / "manifest.json").exists()
    assert snapshot.verified


def test_a_snapshot_is_verified_before_it_is_returned(store):
    snapshot = store.create("test")
    assert store.verify(snapshot.id).ok


def test_adapter_symlinks_survive_the_round_trip(store, populated):
    snapshot = store.create("test")
    link = snapshot.path / "adapters" / "coding" / "active"
    assert link.is_symlink() and os.readlink(link) == "v002"


def test_the_database_copy_is_consistent_and_readable(store, db):
    from gustav.store.db import open_db

    snapshot = store.create("test")
    copy = open_db(snapshot.path / "database" / db.path.name)
    try:
        assert copy.scalar("SELECT COUNT(*) FROM events WHERE kind = 'test'") == 1
    finally:
        copy.close()


def test_a_dry_run_copies_nothing(store):
    before = len(store.list())
    snapshot = store.create("test", dry_run=True)
    assert snapshot.meta["dry_run"]
    assert len(store.list()) == before


def test_the_plan_reports_sizes_and_free_space(store):
    plan = store.plan("test")
    assert plan.estimated_mb > 0
    assert plan.free_mb > 0
    assert any("adapters" == name for name, _, _ in plan.sources)
    assert "would go to" in plan.describe()[0]


def test_a_snapshot_is_refused_when_the_disk_is_nearly_full(populated, db, events):
    store = SnapshotStore(
        populated, location=populated.root / "backups", db=db, events=events,
        min_free_mb=10**9,  # ask for a petabyte
    )
    with pytest.raises(NotEnoughSpace, match="refusing rather than filling"):
        store.create("test")
    assert store.list() == []


def test_a_failure_mid_copy_leaves_no_half_snapshot(store, monkeypatch):
    import gustav.backup.snapshot as module

    real = module.shutil.copytree
    calls = {"n": 0}

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk went away")
        return real(*args, **kwargs)

    monkeypatch.setattr(module.shutil, "copytree", explode)
    with pytest.raises(OSError):
        store.create("test")
    assert store.list() == [], "a partial snapshot must never be left behind"


def test_an_incomplete_snapshot_is_reported_as_such(store):
    snapshot = store.create("test")
    (snapshot.path / "manifest.json").unlink()
    reloaded = store.get(snapshot.id)
    assert not reloaded.complete
    assert not store.verify(snapshot.id).ok


def test_listing_is_chronological(store):
    first = store.create("one")
    second = store.create("two")
    assert [s.id for s in store.list()] == sorted([first.id, second.id])


def test_an_unknown_snapshot_names_what_exists(store):
    store.create("test")
    with pytest.raises(BackupError, match="available"):
        store.get("20000101T000000-nope")


def test_tampering_is_detected_by_verify(store):
    snapshot = store.create("test")
    target = next(
        p for p in (snapshot.path / "adapters").rglob("*") if p.is_file() and p.name != "meta.json"
    )
    target.write_bytes(b"corrupted")
    result = store.verify(snapshot.id)
    assert not result.ok
    assert any(p.kind in ("size", "hash") for p in result.problems)


def test_an_extra_file_is_noticed_when_asked(store):
    snapshot = store.create("test")
    (snapshot.path / "surprise.txt").write_text("hello", encoding="utf-8")
    assert store.verify(snapshot.id).ok, "an extra file is not corruption by default"
    assert not store.verify(snapshot.id, check_extra=True).ok


# --- restore and undo-restore ---------------------------------------------
@pytest.fixture
def restorer(store, events):
    return Restorer(store, events=events)


def test_restore_puts_the_adapters_back(store, restorer, paths):
    snapshot = store.create("before")
    shutil.rmtree(paths.adapters / "coding")
    assert not (paths.adapters / "coding").exists()

    result = restorer.restore(snapshot.id)
    assert "adapters" in result.restored
    assert (paths.adapters / "coding" / "v002" / "adapter_config.json").exists()
    assert os.readlink(paths.adapters / "coding" / "active") == "v002"


def test_restore_snapshots_the_current_state_first(store, restorer, paths):
    """"I restored the wrong one" has to be recoverable."""
    snapshot = store.create("before")
    write_fake_adapter(paths.adapters, "coding", "v003", base_model="qwen3-4b")
    atomic_symlink(paths.adapters / "coding" / "active", "v003")

    result = restorer.restore(snapshot.id)
    assert result.undo_snapshot_id
    assert os.readlink(paths.adapters / "coding" / "active") == "v002"

    # Undo it: v003 comes back.
    undo = restorer.restore(result.undo_snapshot_id)
    assert os.readlink(paths.adapters / "coding" / "active") == "v003"
    assert undo.undo_snapshot_id != result.undo_snapshot_id


def test_restore_of_a_single_module_leaves_the_others_alone(store, restorer, paths):
    snapshot = store.create("before")
    shutil.rmtree(paths.adapters / "coding")
    write_fake_adapter(paths.adapters, "rust", "v002", base_model="qwen3-4b")

    restorer.restore(snapshot.id, module="coding")
    assert (paths.adapters / "coding" / "v002").exists()
    # rust was not in the selection, so the newer version is still there.
    assert (paths.adapters / "rust" / "v002").exists()


def test_restore_of_named_sources_only(store, restorer, paths):
    snapshot = store.create("before")
    paths.registry.write_text("modules: []\n", encoding="utf-8")
    shutil.rmtree(paths.adapters / "coding")

    result = restorer.restore(snapshot.id, only=("registry",))
    assert result.restored == ["registry"]
    assert "name: coding" in paths.registry.read_text()
    assert not (paths.adapters / "coding").exists(), "adapters were not asked for"


def test_a_dry_run_restore_changes_nothing(store, restorer, paths):
    snapshot = store.create("before")
    shutil.rmtree(paths.adapters / "coding")
    result = restorer.restore(snapshot.id, dry_run=True)
    assert result.dry_run and "adapters" in result.restored
    assert not (paths.adapters / "coding").exists()
    assert result.undo_snapshot_id is None


def test_restoring_from_a_corrupted_snapshot_is_refused(store, restorer, paths):
    snapshot = store.create("before")
    target = next(p for p in (snapshot.path / "adapters").rglob("*") if p.is_file())
    target.write_bytes(b"corrupted")
    with pytest.raises(BackupError, match="does not verify"):
        restorer.restore(snapshot.id)


def test_restoring_from_an_incomplete_snapshot_is_refused(store, restorer):
    snapshot = store.create("before")
    (snapshot.path / "manifest.json").unlink()
    with pytest.raises(BackupError, match="incomplete"):
        restorer.restore(snapshot.id)


def test_restoring_something_the_snapshot_lacks_is_an_error(store, restorer):
    snapshot = store.create("before")
    with pytest.raises(BackupError, match="nothing to restore"):
        restorer.restore(snapshot.id, module="no-such-module")


def test_a_failed_swap_leaves_the_original_in_place(store, restorer, paths, monkeypatch):
    """The dangerous moment: the old tree is aside and the new one is not in yet."""
    import gustav.backup.atomic as atomic

    snapshot = store.create("before")
    original = os.readlink(paths.adapters / "coding" / "active")
    real_replace = atomic.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # the second half of the swap
            raise OSError("interrupted")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    with pytest.raises(OSError):
        restorer.restore(snapshot.id, snapshot_first=False)
    monkeypatch.undo()

    assert (paths.adapters / "coding").exists()
    assert os.readlink(paths.adapters / "coding" / "active") == original


# --- retention ------------------------------------------------------------
def fake_snapshot(snapshot_id: str, when: datetime, *, complete: bool = True) -> Snapshot:
    return Snapshot(
        id=snapshot_id, path=Path("/tmp") / snapshot_id, operation="test",
        created_at=when.isoformat(timespec="seconds"), size_bytes=1024, file_count=1,
        complete=complete,
    )


def test_the_last_n_are_always_kept():
    now = datetime(2026, 6, 1, 12, 0)
    snapshots = [fake_snapshot(f"s{i}", now - timedelta(hours=i)) for i in range(10)]
    decision = plan_retention(snapshots, RetentionPolicy(keep_last=3, daily_days=0,
                                                         weekly_weeks=0), now=now)
    assert {s.id for s in decision.keep} >= {"s0", "s1", "s2"}
    assert "s9" in {s.id for s in decision.delete}


def test_one_per_day_is_kept_for_the_daily_window():
    now = datetime(2026, 6, 10, 12, 0)
    snapshots = []
    for day in range(1, 6):
        for hour in (9, 13, 20):
            snapshots.append(
                fake_snapshot(f"d{day}h{hour}", now - timedelta(days=day, hours=hour - 12))
            )
    decision = plan_retention(
        snapshots, RetentionPolicy(keep_last=1, daily_days=7, weekly_weeks=0), now=now
    )
    kept_days = {s.created_at[:10] for s in decision.keep}
    assert len(kept_days) >= 5, "one snapshot per day should survive"
    assert decision.delete, "the other two per day should not"


def test_one_per_week_is_kept_for_the_weekly_window():
    now = datetime(2026, 6, 30, 12, 0)
    snapshots = [
        fake_snapshot(f"w{week}", now - timedelta(weeks=week)) for week in range(1, 5)
    ]
    decision = plan_retention(
        snapshots, RetentionPolicy(keep_last=1, daily_days=0, weekly_weeks=4), now=now
    )
    assert len(decision.keep) >= 4


def test_the_newest_complete_snapshot_is_never_deleted():
    now = datetime(2026, 6, 1, 12, 0)
    snapshots = [fake_snapshot(f"s{i}", now - timedelta(days=400 + i)) for i in range(3)]
    decision = plan_retention(
        snapshots, RetentionPolicy(keep_last=0, daily_days=0, weekly_weeks=0), now=now
    )
    assert "s0" in {s.id for s in decision.keep}
    assert decision.reasons["s0"].startswith("the newest")


def test_a_protected_snapshot_is_never_deleted():
    now = datetime(2026, 6, 1, 12, 0)
    snapshots = [fake_snapshot(f"s{i}", now - timedelta(days=400 + i)) for i in range(4)]
    decision = plan_retention(
        snapshots,
        RetentionPolicy(keep_last=0, daily_days=0, weekly_weeks=0),
        protected={"s3"},
        now=now,
    )
    assert "s3" in {s.id for s in decision.keep}
    assert "approval" in decision.reasons["s3"]


def test_a_stale_incomplete_snapshot_is_cleaned_up():
    now = datetime(2026, 6, 1, 12, 0)
    snapshots = [
        fake_snapshot("good", now - timedelta(hours=1)),
        fake_snapshot("stale", now - timedelta(days=2), complete=False),
        fake_snapshot("fresh", now - timedelta(minutes=5), complete=False),
    ]
    decision = plan_retention(snapshots, RetentionPolicy(keep_last=1), now=now)
    assert "stale" in {s.id for s in decision.delete}
    assert "fresh" in {s.id for s in decision.keep}


def test_retention_on_an_empty_location_does_nothing():
    decision = plan_retention([], RetentionPolicy(), now=datetime.now())
    assert decision.keep == [] and decision.delete == []


def test_the_policy_describes_itself():
    assert "last 5" in RetentionPolicy().describe()


# --- the guard: the choke point -------------------------------------------
@pytest.fixture
def weight_guard(store, db, events):
    return WeightChangeGuard(store, db=db, events=events)


def test_a_guarded_change_takes_a_verified_snapshot_first(weight_guard, store, paths):
    with weight_guard.change("promote coding v003", sources=("adapters",)) as change:
        assert change.snapshot is not None and change.snapshot.verified
        write_fake_adapter(paths.adapters, "coding", "v003", base_model="qwen3-4b")
        change.activate(paths.adapters / "coding" / "active", "v003")
    assert os.readlink(paths.adapters / "coding" / "active") == "v003"
    assert len(store.list()) == 1


def test_the_change_is_logged_with_its_snapshot(weight_guard, db):
    with weight_guard.change("promote coding v003") as change:
        snapshot_id = change.snapshot_id
    row = db.query_one("SELECT * FROM weight_changes ORDER BY id DESC LIMIT 1")
    assert row["status"] == "done"
    assert row["snapshot_id"] == snapshot_id
    assert row["operation"] == "promote coding v003"
    assert row["finished_at"]


def test_a_failing_change_is_rolled_back(weight_guard, paths):
    before = os.readlink(paths.adapters / "coding" / "active")
    with pytest.raises(RuntimeError, match="training diverged"):
        with weight_guard.change("promote coding v003", sources=("adapters",)) as change:
            write_fake_adapter(paths.adapters, "coding", "v003", base_model="qwen3-4b")
            change.activate(paths.adapters / "coding" / "active", "v003")
            raise RuntimeError("training diverged")

    # The symlink is back and the half-made version is gone.
    assert os.readlink(paths.adapters / "coding" / "active") == before
    assert not (paths.adapters / "coding" / "v003").exists()


def test_a_rollback_is_recorded(weight_guard, db):
    with pytest.raises(RuntimeError):
        with weight_guard.change("bad change"):
            raise RuntimeError("nope")
    row = db.query_one("SELECT * FROM weight_changes ORDER BY id DESC LIMIT 1")
    assert row["status"] == "rolled_back"
    assert "nope" in row["detail"]


def test_the_operation_is_refused_when_the_backup_fails(populated, db, events, paths):
    """Section 9: if the backup fails, the operation does not happen."""
    store = SnapshotStore(
        populated, location=populated.root / "backups", db=db, events=events,
        min_free_mb=10**9,
    )
    guard = WeightChangeGuard(store, db=db, events=events)
    before = os.readlink(paths.adapters / "coding" / "active")

    entered = False
    with pytest.raises(BackupRequired, match="Nothing was changed"):
        with guard.change("promote coding v003"):
            entered = True

    assert not entered, "the body must never run without a backup"
    assert os.readlink(paths.adapters / "coding" / "active") == before
    row = db.query_one("SELECT * FROM weight_changes ORDER BY id DESC LIMIT 1")
    assert row["status"] == "refused"
    assert any(e["kind"] == "weights.refused" for e in events.recent())


def test_a_dry_run_change_takes_no_snapshot_and_promotes_nothing(weight_guard, store, paths):
    with weight_guard.change("promote coding v003", dry_run=True) as change:
        change.activate(paths.adapters / "coding" / "active", "v003")
    assert os.readlink(paths.adapters / "coding" / "active") == "v002"
    assert store.list() == []


def test_the_staging_directory_is_cleaned_up(weight_guard, paths):
    with weight_guard.change("promote coding v003") as change:
        (change.staging / "work-in-progress").write_text("x", encoding="utf-8")
        staging = change.staging
    assert not staging.exists()


def test_last_backup_is_recorded_for_status(weight_guard, db):
    with weight_guard.change("promote coding v003") as change:
        pass
    last = db.get_meta("last_backup")
    assert last["id"] == change.snapshot_id and last["verified"]


def test_an_interrupted_change_is_rolled_back_at_startup(store, db, events, paths):
    """Simulates kill -9 between taking the snapshot and finishing the promotion."""
    guard = WeightChangeGuard(store, db=db, events=events)
    snapshot = store.create("promote coding v003")
    db.insert(
        "weight_changes",
        {"operation": "promote coding v003", "snapshot_id": snapshot.id,
         "sources": "adapters", "status": "in_progress"},
    )
    # The crash left a half-made version and a moved symlink behind.
    write_fake_adapter(paths.adapters, "coding", "v003", base_model="qwen3-4b")
    atomic_symlink(paths.adapters / "coding" / "active", "v003")

    undone = guard.rollback_interrupted()
    assert len(undone) == 1 and undone[0]["restored"]
    assert os.readlink(paths.adapters / "coding" / "active") == "v002"
    assert not (paths.adapters / "coding" / "v003").exists()
    row = db.query_one("SELECT status FROM weight_changes ORDER BY id DESC LIMIT 1")
    assert row["status"] == "rolled_back"


def test_an_interrupted_change_without_a_snapshot_is_reported(store, db, events):
    guard = WeightChangeGuard(store, db=db, events=events)
    db.insert(
        "weight_changes",
        {"operation": "mystery", "snapshot_id": None, "status": "in_progress"},
    )
    undone = guard.rollback_interrupted()
    assert not undone[0]["restored"]
    assert "no snapshot" in undone[0]["detail"]


def test_nothing_to_recover_is_the_normal_case(weight_guard):
    assert weight_guard.rollback_interrupted() == []


def test_rollback_step_finds_the_snapshot_a_step_took(store, db, events, paths):
    guard = WeightChangeGuard(store, db=db, events=events)
    with guard.change("train coding", sources=("adapters",), run_id=7, step="training"):
        write_fake_adapter(paths.adapters, "coding", "v003", base_model="qwen3-4b")
    shutil.rmtree(paths.adapters / "coding")

    result = guard.rollback_step(7, "training")
    assert result is not None and "adapters" in result.restored
    assert (paths.adapters / "coding" / "v002").exists()


def test_rollback_step_for_a_step_that_changed_nothing_is_none(weight_guard):
    assert weight_guard.rollback_step(99, "training") is None


def test_disabling_backups_is_allowed_but_announced(store, db, events, paths, caplog):
    guard = WeightChangeGuard(store, db=db, events=events, enabled=False)
    with caplog.at_level("WARNING"):
        with guard.change("promote coding v003") as change:
            assert change.snapshot is None
    assert store.list() == []
    assert any("unprotected" in record.message for record in caplog.records)


def test_the_history_is_readable(weight_guard):
    with weight_guard.change("one"):
        pass
    with weight_guard.change("two"):
        pass
    history = weight_guard.history()
    assert [row["operation"] for row in history] == ["two", "one"]
