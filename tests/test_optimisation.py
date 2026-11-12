"""Self-optimisation: measured trials, inside `tunable` and nowhere else."""

from __future__ import annotations

import pytest

from gustav.config.load import (
    load_config,
    pinned_tunables,
    read_learned_tunable,
    write_learned_tunable,
)
from gustav.config.schema import Config, OptimisationConfig
from gustav.optimize.tuner import Measurement, Tuner


@pytest.fixture
def tuner_for(paths, db, events):
    def build(*, settings=("prefix_cache",), pinned=(), **config_kwargs):
        config = OptimisationConfig(settings=list(settings), **config_kwargs)
        tunable = Config().tunable
        return Tuner(
            config=config,
            tunable=tunable,
            paths=paths,
            db=db,
            events=events,
            pinned=set(pinned),
        )

    return build


def measurer(values: dict[str, float], *, quality: dict[str, float] | None = None):
    """A measurement function keyed by the override it is given."""

    def measure(overrides):
        key = "baseline" if not overrides else next(
            f"{k}={v!r}" for k, v in overrides.items()
        )
        if key not in values:
            raise AssertionError(f"unexpected trial {key}")
        return Measurement(
            tokens_per_second=values[key],
            quality=(quality or {}).get(key),
        )

    return measure


# --- the learned file ---------------------------------------------------
def test_what_you_pin_wins_over_what_it_learns(tmp_path):
    root = tmp_path / "project"
    (root / "var").mkdir(parents=True)
    (root / "config.yaml").write_text(
        "tunable:\n  prefix_cache: true\n", encoding="utf-8"
    )
    write_learned_tunable(root / "var", {"prefix_cache": False, "warm_cache_mb": 2048})

    config, _paths = load_config(root=root)
    assert config.tunable.prefix_cache is True, "the user's value survives"
    assert config.tunable.warm_cache_mb == 2048, "and the learned one applies elsewhere"
    assert pinned_tunables(root / "config.yaml") == {"prefix_cache"}


def test_the_learned_file_is_self_describing(paths):
    path = write_learned_tunable(paths.data_dir, {"prefix_cache": False})
    text = path.read_text(encoding="utf-8")
    assert "self-optimisation" in text
    assert "config.yaml wins" in text
    assert read_learned_tunable(paths.data_dir) == {"prefix_cache": False}


# --- choosing trials ----------------------------------------------------
def test_a_pinned_setting_is_never_tried(tuner_for):
    tuner = tuner_for(settings=("prefix_cache",), pinned=("prefix_cache",))
    assert tuner.candidates() == []
    assert tuner.why_skipped() == ["prefix_cache: pinned in config.yaml"]


def test_the_value_already_in_use_is_not_a_trial(tuner_for):
    tuner = tuner_for()
    assert ("prefix_cache", True) not in tuner.candidates()
    assert ("prefix_cache", False) in tuner.candidates()


def test_a_setting_outside_the_allow_list_is_not_tried(tuner_for):
    tuner = tuner_for(settings=("router_threshold",))
    assert tuner.candidates() == [], "router_threshold is not a speed knob"


def test_a_value_that_already_lost_is_not_tried_again(tuner_for, db):
    tuner = tuner_for()
    db.insert(
        "optimisations",
        {"setting": "prefix_cache", "candidate": repr(False), "status": "reverted"},
    )
    assert tuner.candidates() == []


# --- the loop -----------------------------------------------------------
def test_a_clear_win_is_kept_and_written_down(tuner_for, paths, db):
    tuner = tuner_for(min_relative_gain=0.05)
    result = tuner.run(measurer({"baseline": 40.0, "prefix_cache=False": 60.0}))

    assert result.kept == {"prefix_cache": False}
    assert tuner.tunable.prefix_cache is False
    assert read_learned_tunable(paths.data_dir) == {"prefix_cache": False}
    row = db.query_one("SELECT * FROM optimisations ORDER BY id DESC")
    assert row["status"] == "kept" and row["delta"] == pytest.approx(0.5)


def test_a_small_win_is_noise_and_is_put_back(tuner_for, paths):
    tuner = tuner_for(min_relative_gain=0.05)
    result = tuner.run(measurer({"baseline": 40.0, "prefix_cache=False": 40.8}))

    assert result.kept == {}
    assert tuner.tunable.prefix_cache is True, "the value in use did not change"
    assert read_learned_tunable(paths.data_dir) == {}
    assert "below the 5" in result.trials[0].detail


def test_speed_bought_with_quality_is_refused(tuner_for, paths):
    tuner = tuner_for(min_relative_gain=0.05, max_quality_loss=0.02)
    result = tuner.run(
        measurer(
            {"baseline": 40.0, "prefix_cache=False": 80.0},
            quality={"baseline": 0.8, "prefix_cache=False": 0.5},
        )
    )
    assert result.kept == {}
    assert "worse on the held-out suite" in result.trials[0].detail
    assert read_learned_tunable(paths.data_dir) == {}


def test_a_failed_measurement_is_recorded_and_the_night_goes_on(tuner_for, db):
    def explode(overrides):
        if overrides:
            raise RuntimeError("the model would not load")
        return Measurement(tokens_per_second=40.0)

    result = tuner_for().run(explode)
    assert result.trials[0].status == "failed"
    assert db.query_one("SELECT status FROM optimisations ORDER BY id DESC")["status"] == (
        "failed"
    )


def test_a_failed_baseline_means_no_trials_at_all(tuner_for, db):
    def explode(_overrides):
        raise RuntimeError("no GPU")

    result = tuner_for().run(explode)
    assert result.trials == []
    assert any("baseline measurement failed" in reason for reason in result.skipped)
    assert db.query_one("SELECT * FROM optimisations") is None


def test_later_trials_are_measured_against_what_is_now_in_force(tuner_for, paths):
    """Keeping a change moves the baseline, or the second trial measures the wrong thing."""
    tuner = tuner_for(settings=("prefix_cache", "warm_cache_mb"), max_trials_per_run=2,
                      min_relative_gain=0.05)
    seen: list[float | None] = []

    def measure(overrides):
        if not overrides:
            seen.append(None)
            return Measurement(tokens_per_second=40.0)
        if "prefix_cache" in overrides:
            return Measurement(tokens_per_second=60.0)
        seen.append(60.0)
        return Measurement(tokens_per_second=62.0)

    result = tuner.run(measure)
    assert result.kept == {"prefix_cache": False}, result.summary()
    warm = next(t for t in result.trials if t.setting == "warm_cache_mb")
    assert warm.baseline.tokens_per_second == 60.0
    assert warm.status == "reverted", "62 against 60 is not a 5% win"


def test_a_disabled_optimiser_does_nothing(tuner_for):
    tuner = tuner_for(enabled=False)
    result = tuner.run(measurer({"baseline": 40.0}))
    assert result.trials == [] and result.baseline is None


# --- taking it back -----------------------------------------------------
def test_the_last_kept_change_can_be_put_back(tuner_for, paths, db):
    tuner = tuner_for()
    tuner.run(measurer({"baseline": 40.0, "prefix_cache=False": 80.0}))
    assert read_learned_tunable(paths.data_dir) == {"prefix_cache": False}

    result = tuner.revert_last()
    assert result["setting"] == "prefix_cache"
    assert read_learned_tunable(paths.data_dir) == {}
    assert tuner.tunable.prefix_cache is True, "the live value went back as well"
    assert db.query_one("SELECT status FROM optimisations ORDER BY id DESC")["status"] == (
        "reverted"
    )
    # And it will not be tried again, because it is now a recorded loser.
    assert tuner.candidates() == []


def test_reverting_with_nothing_kept_says_so(tuner_for):
    with pytest.raises(KeyError, match="has not kept any change"):
        tuner_for().revert_last()
