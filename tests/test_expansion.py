"""Self-expansion: the measurements, the budget, and what it is allowed to do alone."""

from __future__ import annotations

import json
import shutil

import pytest

from gustav.config.schema import Config, ExpansionConfig
from gustav.expansion.actions import Expander
from gustav.expansion.proposals import propose
from gustav.expansion.triggers import cluster_requests, measure
from gustav.hw.planner import ExpansionBudget
from gustav.inference.adapters import AdapterStore
from gustav.inference.registry import load_registry
from gustav.sleep.stages import expansion as expansion_step
from gustav.testing.steps import ScriptedModel, make_env, store_plan, write_adapter


def body(module):
    return module.run_step.__wrapped__


@pytest.fixture
def registry(project, repo_root):
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    return load_registry(project / "modules" / "registry.yaml")


@pytest.fixture
def budget():
    return ExpansionBudget(
        max_adapters=6,
        max_lora_rank=32,
        max_total_adapter_params=200_000_000,
        max_adapter_vram_mb=1024,
        warm_cache_mb=8192,
        cold_storage_mb=65536,
    )


@pytest.fixture
def expansion_config():
    return ExpansionConfig(min_uncovered_requests=4, require_approval=["merge_to_base"])


def seed_uncovered(db, requests: list[str]) -> None:
    session = db.insert("sessions", {"source": "test"})
    for text in requests:
        turn_id = db.insert(
            "turns", {"session_id": session, "role": "user", "content": text}
        )
        db.insert(
            "telemetry",
            {
                "metric": "router.decision",
                "value": 0.1,
                "tags": json.dumps({"uncovered": True, "turn_id": turn_id, "modules": []}),
            },
        )


# --- clustering ---------------------------------------------------------
def test_requests_cluster_around_the_word_they_share(registry):
    requests = [
        "how do I write a kubernetes deployment manifest",
        "my kubernetes pod keeps restarting, why",
        "what does a kubernetes service do",
        "what is the capital of france",
    ]
    clusters = cluster_requests(requests, registry, min_share=0.4)
    assert clusters, "three requests about one subject is a cluster"
    name, keywords, members = clusters[0]
    assert name == "kubernetes"
    assert len(members) == 3
    assert "kubernetes" in keywords


def test_words_an_existing_module_already_covers_do_not_name_a_cluster(registry):
    # "rust" and "cargo" are keywords of the rust module, so they cannot name a new one.
    requests = ["rust cargo question one", "rust cargo question two", "rust cargo three"]
    clusters = cluster_requests(requests, registry, min_share=0.4)
    assert all(name not in ("rust", "cargo") for name, _, _ in clusters)


# --- triggers -----------------------------------------------------------
def test_enough_uncovered_requests_about_one_thing_is_a_trigger(db, registry, expansion_config):
    seed_uncovered(db, [f"a kubernetes question number {i}" for i in range(5)])
    triggers = measure(db, registry, expansion_config)
    assert [t.kind for t in triggers] == ["uncovered_topics"]
    assert triggers[0].subject == "kubernetes"
    assert triggers[0].evidence["requests"] == 5


def test_a_few_uncovered_requests_are_a_bad_day_not_a_trigger(db, registry, expansion_config):
    seed_uncovered(db, ["one kubernetes question", "another kubernetes question"])
    assert measure(db, registry, expansion_config) == []


def test_a_module_whose_gate_results_stopped_moving_is_saturated(db, registry, expansion_config):
    for _ in range(3):
        db.insert(
            "gate_results",
            {"module": "rust", "suite": "heldout", "delta": 0.001, "verdict": "fail",
             "items": 20},
        )
    triggers = [t for t in measure(db, registry, expansion_config) if t.kind == "saturated_module"]
    assert triggers and triggers[0].subject == "rust"
    assert triggers[0].evidence["deltas"] == [0.001, 0.001, 0.001]


def test_a_module_that_is_still_improving_is_not_saturated(db, registry, expansion_config):
    for delta in (0.05, 0.04, 0.03):
        db.insert(
            "gate_results",
            {"module": "rust", "suite": "heldout", "delta": delta, "verdict": "pass",
             "items": 20},
        )
    assert not [
        t for t in measure(db, registry, expansion_config) if t.kind == "saturated_module"
    ]


def test_an_adapter_nobody_uses_is_a_pruning_candidate(db, registry, paths, expansion_config):
    write_adapter(paths.adapters / "rust" / "v001")
    adapters = AdapterStore(paths.adapters)
    triggers = [
        t for t in measure(db, registry, expansion_config, adapters=adapters)
        if t.kind == "unused_module"
    ]
    assert triggers and triggers[0].subject == "rust"


def test_an_always_on_module_is_never_a_pruning_candidate(db, registry, paths, expansion_config):
    write_adapter(paths.adapters / "general" / "v001")
    adapters = AdapterStore(paths.adapters)
    assert not [
        t for t in measure(db, registry, expansion_config, adapters=adapters)
        if t.kind == "unused_module" and t.subject == "general"
    ]


def test_a_repeatedly_promoted_always_on_module_is_a_merge_candidate(
    db, registry, paths, expansion_config
):
    write_adapter(paths.adapters / "general" / "v001")
    for _ in range(3):
        db.insert("training_runs", {"module": "general", "version": "v001",
                                    "status": "promoted"})
    triggers = [
        t for t in measure(db, registry, expansion_config, adapters=AdapterStore(paths.adapters))
        if t.kind == "merge_candidate"
    ]
    assert triggers and triggers[0].subject == "general"


# --- proposals and the budget -------------------------------------------
def test_a_new_module_is_proposed_with_keywords_the_router_can_match(
    db, registry, budget, expansion_config
):
    seed_uncovered(db, [f"a kubernetes manifest question {i}" for i in range(5)])
    triggers = measure(db, registry, expansion_config)
    proposals = propose(
        triggers, registry=registry, budget=budget, config=expansion_config,
        plan_rank=16, adapter_count=2,
    )
    assert [p.kind for p in proposals] == ["new_module"]
    proposal = proposals[0]
    assert proposal.applicable and not proposal.needs_approval
    assert proposal.payload["name"] == "kubernetes"
    assert "kubernetes" in proposal.payload["keywords"]
    assert "kubernetes" in proposal.payload["description"].lower()


def test_the_adapter_budget_refuses_a_module_that_does_not_fit(
    db, registry, expansion_config
):
    seed_uncovered(db, [f"a kubernetes question {i}" for i in range(5)])
    tight = ExpansionBudget(
        max_adapters=2, max_lora_rank=16, max_total_adapter_params=10_000_000,
        max_adapter_vram_mb=64, warm_cache_mb=512, cold_storage_mb=1024,
    )
    proposals = propose(
        measure(db, registry, expansion_config), registry=registry, budget=tight,
        config=expansion_config, plan_rank=16, adapter_count=2,
    )
    assert not proposals[0].applicable
    assert "2 adapters" in proposals[0].refused


def test_a_rank_rise_is_capped_by_what_the_card_can_hold(db, registry, expansion_config):
    for _ in range(3):
        db.insert("gate_results", {"module": "rust", "suite": "heldout", "delta": 0.0,
                                   "verdict": "fail", "items": 20})
    tight = ExpansionBudget(
        max_adapters=6, max_lora_rank=16, max_total_adapter_params=200_000_000,
        max_adapter_vram_mb=1024, warm_cache_mb=8192, cold_storage_mb=65536,
    )
    proposals = propose(
        measure(db, registry, expansion_config), registry=registry, budget=tight,
        config=expansion_config, plan_rank=16, adapter_count=1,
    )
    rank = next(p for p in proposals if p.kind == "raise_rank")
    assert not rank.applicable and "above the 16" in rank.refused


def test_a_base_merge_always_needs_approval_even_if_configured_otherwise(
    db, registry, budget, paths
):
    write_adapter(paths.adapters / "general" / "v001")
    for _ in range(3):
        db.insert("training_runs", {"module": "general", "version": "v001",
                                    "status": "promoted"})
    config = ExpansionConfig(allow_base_merge=True, require_approval=[])
    proposals = propose(
        measure(db, registry, config, adapters=AdapterStore(paths.adapters)),
        registry=registry, budget=budget, config=config, plan_rank=16, adapter_count=1,
    )
    merge = next(p for p in proposals if p.kind == "merge_to_base")
    assert merge.needs_approval, "rewriting the base is never automatic"
    assert merge.cost["snapshot_includes_base"] is True


def test_switches_that_are_off_refuse_rather_than_vanish(db, registry, budget):
    seed_uncovered(db, [f"a kubernetes question {i}" for i in range(5)])
    config = ExpansionConfig(min_uncovered_requests=4, allow_new_modules=False)
    proposals = propose(
        measure(db, registry, config), registry=registry, budget=budget, config=config,
        plan_rank=16, adapter_count=1,
    )
    assert not proposals[0].applicable
    assert "allow_new_modules is off" in proposals[0].refused


# --- applying -----------------------------------------------------------
@pytest.fixture
def expander(project, paths, db, events, registry):
    from gustav.backup.guard import WeightChangeGuard
    from gustav.backup.restore import Restorer
    from gustav.backup.snapshot import SnapshotStore

    store = SnapshotStore(
        paths, location=project / "backups", db=db, events=events, min_free_mb=1
    )
    guard = WeightChangeGuard(
        store, restorer=Restorer(store, events=events), db=db, events=events
    )
    config = Config.model_validate({"logging": {"console": False}})
    return Expander(
        paths=paths, config=config, db=db, events=events, guard=guard,
        adapters=AdapterStore(paths.adapters),
    )


def proposal_for(kind: str, subject: str, **payload):
    from gustav.expansion.proposals import Proposal

    return Proposal(
        kind=kind,
        subject=subject,
        rationale=f"because of {kind}",
        payload=payload,
        needs_approval=False,
    )


def test_a_new_module_is_written_to_the_registry_behind_a_snapshot(expander, paths, db):
    applied = expander.apply(
        proposal_for("new_module", "kubernetes", name="kubernetes",
                     description="Kubernetes manifests and pods",
                     keywords=["kubernetes", "pod", "manifest"], lora_rank=16)
    )
    assert applied.ok, applied.detail
    assert applied.snapshot_id, "a verified snapshot was taken first"

    registry = load_registry(paths.registry)
    assert "kubernetes" in registry
    assert registry.get("kubernetes").keywords == ("kubernetes", "pod", "manifest")
    assert registry.get("kubernetes").lora_rank == 16
    row = db.query_one("SELECT * FROM expansion_log ORDER BY id DESC")
    assert row["status"] == "applied" and row["snapshot_id"] == applied.snapshot_id


def test_a_new_module_has_no_adapter_until_a_night_trains_one(expander, paths):
    expander.apply(
        proposal_for("new_module", "kubernetes", name="kubernetes", description="k8s",
                     keywords=["kubernetes"], lora_rank=16)
    )
    assert AdapterStore(paths.adapters).versions("kubernetes") == []


def test_raising_a_rank_changes_only_that_module(expander, paths):
    applied = expander.apply(
        proposal_for("raise_rank", "rust", module="rust", lora_rank=24, previous=16)
    )
    assert applied.ok
    registry = load_registry(paths.registry)
    assert registry.get("rust").lora_rank == 24
    assert registry.get("coding").lora_rank == 0, "untouched modules keep the plan default"


def test_pruning_archives_the_adapter_instead_of_deleting_it(expander, paths):
    write_adapter(paths.adapters / "rust" / "v001")
    applied = expander.apply(proposal_for("prune_module", "rust", module="rust"))
    assert applied.ok, applied.detail
    assert "rust" not in load_registry(paths.registry)
    assert not (paths.adapters / "rust").exists()
    archived = list((paths.data_dir / "archive").glob("rust-*/rust"))
    assert archived and (archived[0] / "v001" / "adapter_model.safetensors").exists()


def test_an_expansion_can_be_taken_back(expander, paths, db):
    expander.apply(
        proposal_for("new_module", "kubernetes", name="kubernetes", description="k8s",
                     keywords=["kubernetes"], lora_rank=16)
    )
    assert "kubernetes" in load_registry(paths.registry)

    result = expander.revert_last()
    assert result["kind"] == "new_module"
    assert "kubernetes" not in load_registry(paths.registry), "the registry went back"
    assert result["undo_snapshot"], "and the revert is itself undoable"
    assert db.query_one("SELECT status FROM expansion_log ORDER BY id DESC")["status"] == (
        "reverted"
    )


def test_reverting_with_no_history_says_so(expander):
    with pytest.raises(KeyError, match="no applied expansion"):
        expander.revert_last()


def test_a_failing_action_is_recorded_and_changes_nothing(expander, paths, db):
    before = load_registry(paths.registry).names
    applied = expander.apply(proposal_for("raise_rank", "nonexistent", module="nonexistent",
                                          lora_rank=24))
    assert not applied.ok
    assert load_registry(paths.registry).names == before
    assert db.query_one("SELECT status FROM expansion_log ORDER BY id DESC")["status"] == (
        "failed"
    )


def test_a_refused_proposal_is_logged_rather_than_attempted(expander, db):
    from gustav.expansion.proposals import Proposal

    refused = Proposal(kind="new_module", subject="x", rationale="r",
                       refused="the budget says no")
    applied = expander.apply(refused)
    assert not applied.ok and applied.detail == "the budget says no"
    assert db.query_one("SELECT status FROM expansion_log ORDER BY id DESC")["status"] == (
        "refused"
    )


def test_an_approval_is_filed_with_its_evidence(expander, db):
    from gustav.expansion.proposals import Proposal

    proposal = Proposal(
        kind="merge_to_base", subject="general", rationale="promoted three times",
        evidence={"promotions": 3}, payload={"module": "general"}, needs_approval=True,
    )
    approval_id = expander.file_approval(proposal, run_id=1)
    row = db.query_one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
    assert row["kind"] == "expansion" and row["status"] == "pending"
    assert json.loads(row["payload"])["evidence"] == {"promotions": 3}
    log = db.query_one("SELECT * FROM expansion_log ORDER BY id DESC")
    assert log["status"] == "pending" and log["approval_id"] == approval_id


# --- the step -----------------------------------------------------------
@pytest.fixture
def step_env(project, paths, db, registry, repo_root):
    def build(**overrides):
        data = {
            "logging": {"console": False},
            "backup": {"location": str(project / "backups"), "min_free_mb": 1},
            "expansion": {
                "min_uncovered_requests": 4,
                "require_approval": ["merge_to_base"],
                # On, so the merge path reaches the approval queue rather than being
                # refused by the switch. It still never applies by itself.
                "allow_base_merge": True,
            },
        }
        data.update(overrides)
        config = Config.model_validate(data)
        env = make_env(config=config, paths=paths, db=db, step="expansion",
                       model=ScriptedModel(available=False))
        return env

    return build


def test_the_step_does_nothing_without_a_plan(step_env):
    result = body(expansion_step)(step_env())
    assert "no hardware plan" in result["reason"]


def test_the_step_applies_a_cheap_change_and_asks_about_the_rest(step_env, db, paths):
    store_plan(db)
    seed_uncovered(db, [f"a kubernetes manifest question {i}" for i in range(5)])
    write_adapter(paths.adapters / "general" / "v001")
    for _ in range(3):
        db.insert("training_runs", {"module": "general", "version": "v001",
                                    "status": "promoted"})

    result = body(expansion_step)(step_env())
    assert len(result["applied"]) == 1
    assert result["applied"][0]["kind"] == "new_module"
    assert "kubernetes" in load_registry(paths.registry)
    # The merge candidate was not done; it was asked about.
    assert [p["kind"] for p in result["pending"]] == ["merge_to_base"]
    assert db.query_one(
        "SELECT COUNT(*) AS n FROM approvals WHERE kind = 'expansion' AND status = 'pending'"
    )["n"] == 1


def test_one_structural_change_per_night(step_env, db, paths):
    store_plan(db)
    seed_uncovered(
        db,
        [f"a kubernetes question {i}" for i in range(5)]
        + [f"a terraform question {i}" for i in range(5)],
    )
    result = body(expansion_step)(step_env())
    assert len(result["applied"]) == 1
    assert any("held back until tomorrow" in line for line in result["refused"])


def test_a_disabled_expansion_measures_nothing(step_env, db):
    store_plan(db)
    seed_uncovered(db, [f"a kubernetes question {i}" for i in range(5)])
    result = body(expansion_step)(step_env(expansion={"enabled": False}))
    assert "enabled is false" in result["reason"]
    assert result["triggers"] == []


def test_a_dry_run_proposes_without_changing_anything(step_env, db, paths):
    store_plan(db)
    seed_uncovered(db, [f"a kubernetes question {i}" for i in range(5)])
    env = step_env()
    env.dry_run = True
    result = body(expansion_step)(env)
    assert result["proposals"] and result["applied"] == []
    assert "kubernetes" not in load_registry(paths.registry)


def test_a_resumed_step_does_not_redo_what_it_already_did(step_env, db, paths):
    """It resumes by name: re-measured triggers may come back in a different order."""
    store_plan(db)
    seed_uncovered(db, [f"a kubernetes question {i}" for i in range(5)])

    first = step_env()
    body(expansion_step)(first)
    assert "kubernetes" in load_registry(paths.registry)
    applied = db.query_one("SELECT COUNT(*) AS n FROM expansion_log WHERE status = 'applied'")
    assert applied["n"] == 1

    # The same night resuming with what it had written down: nothing is done twice.
    checkpoint = first.ctx.checkpoints[-1]
    assert checkpoint["handled"] == ["new_module:kubernetes"]
    resumed = step_env()
    resumed.checkpoint = dict(checkpoint)
    result = body(expansion_step)(resumed)
    assert result["applied"] == []
    assert db.query_one(
        "SELECT COUNT(*) AS n FROM expansion_log WHERE status = 'applied'"
    )["n"] == 1


def test_other_adapters_are_stamped_stale_when_the_base_changes(expander, paths, db, monkeypatch):
    """A merge leaves every other adapter a delta from weights that no longer exist."""
    from gustav.hw.service import META_PLAN

    write_adapter(paths.adapters / "general" / "v003", version="v003")
    (paths.adapters / "general" / "active").symlink_to("v003")
    write_adapter(paths.adapters / "rust" / "v002", version="v002")
    (paths.adapters / "rust" / "active").symlink_to("v002")
    db.set_meta(META_PLAN, {"base_model": "tiny", "compute_dtype": "float32"})
    (paths.models / "tiny").mkdir(parents=True)
    (paths.models / "tiny" / "config.json").write_text("{}", encoding="utf-8")

    # The merge itself needs torch and minutes; the bookkeeping around it is what this
    # checks, so it stands in for the merge. test_merge.py covers the real one.
    def fake_merge(**kwargs):
        from pathlib import Path as _Path

        output = _Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text("{}", encoding="utf-8")
        (output / "model.safetensors").write_bytes(b"\x00")
        return {"size_mb": 1.0, "output": str(output), "weight_files": ["model.safetensors"]}

    monkeypatch.setattr("gustav.expansion.merge.merge_adapter_into_base", fake_merge)
    applied = expander.apply(proposal_for("merge_to_base", "general", module="general"))
    assert applied.ok, applied.detail
    assert "stale" in applied.detail

    store = AdapterStore(paths.adapters)
    assert store.versions("general") == [], "the merged adapter was archived"
    rust = store.get("rust", "v002")
    assert "merged into tiny" in rust.meta["stale_base"]


def test_a_stale_adapter_is_not_loaded(paths, db, events, repo_root):
    """Applying a delta to the wrong weights is worse than serving the base alone."""
    import shutil

    from gustav.config.schema import Config
    from gustav.hw.detect import simulated
    from gustav.hw.ladder import load_ladder
    from gustav.hw.service import HardwareService
    from gustav.inference.harness import Harness
    from gustav.testing.fakes import FakeBackend

    shutil.copy(repo_root / "modules" / "registry.yaml", paths.registry)
    config = Config.model_validate({"logging": {"console": False}})
    hardware = HardwareService(
        config, paths, db, events, detector=simulated("small-12gb"), ladder=load_ladder()
    )
    hardware.ensure_plan()
    write_adapter(paths.adapters / "rust" / "v001", base_model="qwen3-4b")
    (paths.adapters / "rust" / "active").symlink_to("v001")
    store = AdapterStore(paths.adapters)
    meta = dict(store.get("rust", "v001").meta)
    meta["stale_base"] = "coding@v002 was merged into qwen3-4b"
    store.write_meta("rust", "v001", meta)

    harness = Harness(
        config=config, paths=paths, hardware=hardware, events=events, db=db,
        backend=FakeBackend(base_model="qwen3-4b"),
    )
    import asyncio

    asyncio.run(harness.load())
    assert "rust" not in harness.describe()["adapters"]
    assert any("is stale" in note for note in harness.describe()["notes"])
