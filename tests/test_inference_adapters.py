"""The adapter store: versions, atomic promotion and rollback."""

from __future__ import annotations

import json
import os

import pytest

from gustav.inference.adapters import ACTIVE_LINK, AdapterError, AdapterStore
from gustav.testing.fakes import write_fake_adapter


@pytest.fixture
def store(tmp_path):
    return AdapterStore(tmp_path / "adapters")


def add(store: AdapterStore, module: str, version: str, **meta) -> None:
    write_fake_adapter(store.root, module, version, **meta)


def test_an_empty_store_has_nothing():
    assert AdapterStore("/nonexistent").modules() == []


def test_versions_are_listed_in_order(store):
    for version in ("v003", "v001", "v002"):
        add(store, "coding", version)
    assert [v.version for v in store.versions("coding")] == ["v001", "v002", "v003"]


def test_next_version_counts_up(store):
    assert store.next_version("coding") == "v001"
    add(store, "coding", "v001")
    assert store.next_version("coding") == "v002"
    add(store, "coding", "v009")
    assert store.next_version("coding") == "v010"


def test_reserve_creates_an_empty_directory(store):
    version = store.reserve("coding")
    assert version.path.is_dir() and not version.complete


def test_reserving_an_existing_version_is_refused(store):
    add(store, "coding", "v001")
    with pytest.raises(AdapterError, match="already exists"):
        store.reserve("coding", "v001")


def test_activation_points_the_symlink(store):
    add(store, "coding", "v001")
    store.activate("coding", "v001")
    link = store.root / "coding" / ACTIVE_LINK
    assert link.is_symlink() and os.readlink(link) == "v001"
    assert store.active("coding").version == "v001"


def test_an_incomplete_version_cannot_be_activated(store):
    """Half a training run must never become the live adapter."""
    store.reserve("coding", "v001")
    with pytest.raises(AdapterError, match="no adapter weights"):
        store.activate("coding", "v001")


def test_promotion_is_atomic_and_never_leaves_active_missing(store):
    """There is no instant at which `active` does not exist."""
    add(store, "coding", "v001")
    add(store, "coding", "v002")
    store.activate("coding", "v001")
    link = store.root / "coding" / ACTIVE_LINK
    assert link.exists()
    store.activate("coding", "v002")
    assert link.exists() and store.active("coding").version == "v002"
    # No leftover swap file from the rename.
    assert not list(store.root.glob("coding/.active*"))


def test_a_crash_before_the_swap_leaves_the_old_version_active(store):
    """Simulates dying between writing the new weights and switching to them."""
    add(store, "coding", "v001")
    store.activate("coding", "v001")
    add(store, "coding", "v002")  # trained, written, but never activated
    assert store.active("coding").version == "v001"
    assert store.latest("coding").version == "v002"


def test_rollback_goes_to_the_previous_version(store):
    for version in ("v001", "v002", "v003"):
        add(store, "coding", version)
        store.activate("coding", version)
    assert store.rollback("coding").version == "v002"
    assert store.rollback("coding").version == "v001"


def test_rollback_with_nothing_to_roll_back_to_explains_itself(store):
    add(store, "coding", "v001")
    store.activate("coding", "v001")
    with pytest.raises(AdapterError, match="no earlier version"):
        store.rollback("coding")


def test_metadata_round_trips(store):
    add(store, "coding", "v001")
    store.write_meta("coding", "v001", {"base_model": "qwen3-4b", "benchmark_score": 0.72})
    version = store.get("coding", "v001")
    assert version.base_model == "qwen3-4b"
    assert version.meta["benchmark_score"] == 0.72
    assert "0.72" in version.describe()


def test_metadata_is_written_atomically(store):
    add(store, "coding", "v001")
    store.write_meta("coding", "v001", {"a": 1})
    assert not list((store.root / "coding" / "v001").glob("*.tmp"))


def test_unreadable_metadata_does_not_raise(store):
    add(store, "coding", "v001")
    (store.root / "coding" / "v001" / "meta.json").write_text("{not json", encoding="utf-8")
    assert store.get("coding", "v001").meta == {}


def test_discard_removes_a_candidate(store):
    add(store, "coding", "v001")
    add(store, "coding", "v002")
    store.activate("coding", "v001")
    store.discard("coding", "v002")
    assert [v.version for v in store.versions("coding")] == ["v001"]


def test_the_active_version_cannot_be_discarded(store):
    add(store, "coding", "v001")
    store.activate("coding", "v001")
    with pytest.raises(AdapterError, match="it is active"):
        store.discard("coding", "v001")


def test_archive_moves_instead_of_deleting(store, tmp_path):
    add(store, "unused", "v001")
    target = store.archive("unused", tmp_path / "archive")
    assert target.exists() and not (store.root / "unused").exists()
    assert "unused" not in store.modules()


def test_archiving_twice_keeps_both_copies(store, tmp_path):
    add(store, "unused", "v001")
    first = store.archive("unused", tmp_path / "archive")
    add(store, "unused", "v001")
    second = store.archive("unused", tmp_path / "archive")
    assert first.exists() and second.exists() and first != second


def test_describe_summarises_every_module(store):
    add(store, "coding", "v001", base_model="qwen3-4b")
    add(store, "rust", "v001", base_model="qwen3-4b")
    store.activate("coding", "v001")
    described = store.describe()
    assert described["coding"]["active"] == "v001"
    assert described["rust"]["active"] is None
    assert described["coding"]["base_model"] == "qwen3-4b"


def test_a_stray_directory_is_not_mistaken_for_a_version(store):
    add(store, "coding", "v001")
    (store.root / "coding" / "scratch").mkdir()
    assert [v.version for v in store.versions("coding")] == ["v001"]


def test_active_versions_is_a_flat_map(store):
    for module in ("coding", "rust"):
        add(store, module, "v001")
        store.activate(module, "v001")
    assert store.active_versions() == {"coding": "v001", "rust": "v001"}
