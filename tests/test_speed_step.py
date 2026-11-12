"""The speed step: fitting the router, and measuring tunable changes."""

from __future__ import annotations

import shutil

import pytest

from gustav.config.schema import Config
from gustav.learn.examples import Example, ExampleStore
from gustav.optimize.router_fit import RouterModel
from gustav.sleep.stages import speed as speed_step
from gustav.testing.steps import ScriptedModel, make_env, store_plan


def body(module):
    return module.run_step.__wrapped__


@pytest.fixture
def env_for(project, paths, db, repo_root):
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")

    def build(*, model=None, **overrides):
        data = {"logging": {"console": False}, "sleep": {"heldout_fraction": 0.0}}
        data.update(overrides)
        return make_env(
            config=Config.model_validate(data), paths=paths, db=db, step="speed",
            model=model,
        )

    return build


def seed_material(db, *, module="rust", count=30):
    store = ExampleStore(db, heldout_fraction=0.0)
    store.add_many(
        Example(prompt=f"an axum extractor question number {i}", completion="an answer",
                kind="study", module=module)
        for i in range(count)
    )


def seed_heldout(db, count=8):
    store = ExampleStore(db, heldout_fraction=0.0)
    for index in range(count):
        store.add(
            Example(prompt=f"held out question {index}", completion="the expected answer",
                    kind="study", module="general", split="heldout")
        )


# --- the router ---------------------------------------------------------
def test_with_enough_material_a_fitted_router_is_installed(env_for, db, paths):
    seed_material(db)
    result = body(speed_step)(env_for())
    router = result["router"]
    assert router["installed"], router["reason"]
    assert (paths.data_dir / "router" / "model.json").exists()
    model = RouterModel.load(paths.data_dir / "router")
    assert model is not None and model.weights["rust"]


def test_with_too_little_material_the_keyword_scorer_is_kept(env_for, db, paths):
    seed_material(db, count=2)
    result = body(speed_step)(env_for())
    assert not result["router"]["installed"]
    assert "before a fitted router" in result["router"]["reason"]
    assert not (paths.data_dir / "router" / "model.json").exists()


def test_a_dry_run_fits_but_does_not_install(env_for, db, paths):
    seed_material(db)
    env = env_for()
    env.dry_run = True
    result = body(speed_step)(env)
    assert not result["router"]["installed"]
    assert "dry run" in result["router"]["reason"]
    assert not (paths.data_dir / "router" / "model.json").exists()


def test_a_fitted_router_that_disagrees_with_the_keywords_is_not_installed(
    env_for, db, paths, monkeypatch
):
    seed_material(db)
    monkeypatch.setattr(speed_step, "MIN_AGREEMENT", 1.01)
    # Requests the keyword scorer is confident about, so agreement is measurable.
    session = db.insert("sessions", {"source": "test"})
    for index in range(10):
        db.insert("turns", {"session_id": session, "role": "user",
                            "content": f"a question about cargo and clippy {index}"})
    result = body(speed_step)(env_for())
    assert not result["router"]["installed"]
    assert "agrees with the keyword scorer" in result["router"]["reason"]
    assert not (paths.data_dir / "router" / "model.json").exists()


# --- the tunables -------------------------------------------------------
def test_without_a_model_no_trial_is_run(env_for, db):
    seed_material(db)
    result = body(speed_step)(env_for())
    optimisation = result["optimisation"]
    assert optimisation["trials"] == []
    assert any("baseline measurement failed" in reason for reason in optimisation["skipped"])


def test_a_faster_setting_is_kept(env_for, db, paths):
    from gustav.config.load import read_learned_tunable

    model = ScriptedModel(available=True)
    model.speeds = lambda settings: 80.0 if settings["prefix_cache"] is False else 40.0
    result = body(speed_step)(
        env_for(model=model, optimisation={"settings": ["prefix_cache"],
                                           "max_trials_per_run": 1})
    )
    assert result["optimisation"]["kept"] == {"prefix_cache": "False"}
    assert read_learned_tunable(paths.data_dir) == {"prefix_cache": False}


def test_a_setting_that_costs_quality_is_put_back(env_for, db, paths):
    from gustav.config.load import read_learned_tunable

    seed_heldout(db)
    model = ScriptedModel(available=True)
    model.speeds = lambda settings: 80.0 if settings["prefix_cache"] is False else 40.0
    model.qualities = lambda settings: 0.4 if settings["prefix_cache"] is False else 0.9
    result = body(speed_step)(
        env_for(model=model, optimisation={"settings": ["prefix_cache"],
                                           "max_trials_per_run": 1})
    )
    assert result["optimisation"]["kept"] == {}
    assert read_learned_tunable(paths.data_dir) == {}
    assert any("worse on the held-out suite" in trial
               for trial in result["optimisation"]["trials"])


def test_the_configuration_is_left_as_it_was_after_a_losing_trial(env_for, db):
    model = ScriptedModel(available=True)
    model.speeds = lambda settings: 40.0  # no change either way
    env = env_for(model=model, optimisation={"settings": ["prefix_cache"],
                                             "max_trials_per_run": 1})
    before = env.config.tunable.prefix_cache
    body(speed_step)(env)
    assert env.config.tunable.prefix_cache is before


def test_a_pinned_setting_is_reported_as_untouchable(env_for, project, db):
    (project / "config.yaml").write_text(
        "tunable:\n  prefix_cache: true\n", encoding="utf-8"
    )
    model = ScriptedModel(available=True)
    model.speeds = 40.0
    result = body(speed_step)(
        env_for(model=model, optimisation={"settings": ["prefix_cache"]})
    )
    assert result["optimisation"]["skipped"] == ["prefix_cache: pinned in config.yaml"]
    assert result["optimisation"]["trials"] == []


# --- the draft model ----------------------------------------------------
def test_no_draft_model_is_offered_for_a_model_nothing_is_smaller_than(env_for, db):
    store_plan(db, base_model="qwen3-0.6b")
    draft = body(speed_step)(env_for())["draft"]
    assert draft["draft_model"] is None
    assert "small enough" in draft["reason"]


def test_a_draft_model_that_is_not_downloaded_is_reported_not_assumed(env_for, db):
    store_plan(db, speculative_decoding=True)
    draft = body(speed_step)(env_for())["draft"]
    assert draft["wanted"] is True
    assert draft["draft_model"] == "qwen3-0.6b"
    assert draft["on_disk"] is False and draft["supported"] is False
    assert "ai model pull qwen3-0.6b" in draft["reason"]


def test_a_draft_model_on_disk_is_reported_as_available(env_for, db, paths, repo_root):
    """The weights the real project uses are symlinked in, not copied."""
    store_plan(db, speculative_decoding=True)
    source = repo_root / "models" / "qwen3-0.6b"
    if not (source / "config.json").exists():
        pytest.skip("qwen3-0.6b is not downloaded on this machine")
    paths.models.mkdir(parents=True, exist_ok=True)
    (paths.models / "qwen3-0.6b").symlink_to(source)

    draft = body(speed_step)(env_for())["draft"]
    assert draft["on_disk"] and draft["supported"]
    assert "available as a draft model" in draft["reason"]


def test_the_acceptance_rate_is_reported_once_a_benchmark_has_measured_it(env_for, db):
    store_plan(db, speculative_decoding=True)
    db.insert("telemetry", {"metric": "speculation.acceptance", "value": 0.62})
    draft = body(speed_step)(env_for())["draft"]
    assert draft["acceptance"] == 0.62


def test_without_a_plan_the_draft_state_says_so(env_for, db):
    assert "no hardware plan" in body(speed_step)(env_for())["draft"]["reason"]
