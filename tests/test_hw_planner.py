"""The planner, against simulated machines."""

from __future__ import annotations

import yaml

import pytest

from gustav.config.schema import HardwareConfig
from gustav.hw.detect import SimulatedDetector, simulated
from gustav.hw.ladder import load_ladder
from gustav.hw.planner import Plan, Planner, PlannerError
from gustav.hw.speed import METRIC_TOKENS_PER_SECOND, Measurements


@pytest.fixture(scope="module")
def ladder():
    return load_ladder()


def config(**kwargs) -> HardwareConfig:
    return HardwareConfig.model_validate(kwargs)


def planner(preset: str, ladder, **kwargs) -> Planner:
    return Planner(simulated(preset).detect(), config(**kwargs), ladder=ladder)


# --- the three machines from the plan -------------------------------------
def test_12gb_desktop_lands_on_a_4b_model_that_can_train(ladder):
    plan = planner("small-12gb", ladder).plan()
    assert plan.base_model == "qwen3-4b"
    assert plan.quantization == "4bit-nf4"
    assert plan.training_allowed
    assert plan.training_gpus == [0] and plan.inference_gpus == [0]
    assert plan.vram_training_mb < 12288
    assert plan.compute_dtype == "bfloat16"


def test_the_attention_backend_matches_what_is_installed(ladder):
    """The GPU supporting FlashAttention-2 is not the same as having the package."""
    from gustav.hw.detect import flash_attention_available

    plan = planner("small-12gb", ladder).plan()
    expected = "flash_attention_2" if flash_attention_available() else "sdpa"
    assert plan.attention == expected
    if expected == "sdpa":
        assert any("flash-attn package is not installed" in line
                   for line in plan.explanation)


def test_4gb_turing_laptop_gets_fp16_sdpa_and_a_small_model(ladder):
    plan = planner("tiny-4gb", ladder, min_tokens_per_second=10, balance=0.25).plan()
    assert plan.base_model in ("qwen3-0.6b", "qwen3-1.7b")
    assert plan.compute_dtype == "float16", "Turing has no bfloat16"
    assert plan.attention == "sdpa", "FlashAttention-2 needs sm_80+"
    assert plan.vram_inference_mb < 4096


def test_4gb_without_a_training_requirement_serves_a_bigger_model(ladder):
    """The spec's tiny-4gb case: inference only, trained elsewhere."""
    trained_here = planner("tiny-4gb", ladder, min_tokens_per_second=10).plan()
    serving_only = planner(
        "tiny-4gb", ladder, min_tokens_per_second=10, allow_training=False,
        require_training=False, max_seq_len=1024,
    ).plan()
    assert not serving_only.training_allowed
    assert serving_only.training_gpus == []
    rungs = {m.id: m.rung for m in ladder.models}
    assert rungs[serving_only.base_model] >= rungs[trained_here.base_model]
    assert "serves only" in " ".join(serving_only.explanation)


def test_four_card_server_separates_serving_from_training(ladder):
    plan = planner("server-multigpu", ladder, balance=0.8).plan()
    assert plan.inference_gpus == [0]
    assert plan.training_gpus == [1, 2, 3]
    assert plan.training_allowed
    assert "chat stays up during sleep" in " ".join(plan.explanation)


def test_four_card_server_reaches_the_moe_model(ladder):
    plan = planner("server-multigpu", ladder, balance=1.0).plan()
    model = ladder.model(plan.base_model)
    assert model.rung >= ladder.model("qwen3-14b").rung


def test_a_machine_with_no_gpu_is_refused_clearly(ladder):
    with pytest.raises(PlannerError) as exc:
        planner("cpu-only", ladder).plan()
    assert "no GPU was detected" in str(exc.value)
    assert "try:" in str(exc.value)


# --- the balance knob -----------------------------------------------------
def test_balance_moves_the_choice_along_the_ladder(ladder):
    chosen = {}
    for balance in (0.0, 0.5, 1.0):
        plan = planner("small-12gb", ladder, balance=balance).plan()
        chosen[balance] = ladder.model(plan.base_model).rung
    assert chosen[0.0] < chosen[1.0], "balance 0 must pick a smaller, faster model"
    assert chosen[0.0] <= chosen[0.5] <= chosen[1.0]


def test_balance_works_by_raising_the_speed_floor(ladder):
    fast = planner("small-12gb", ladder, balance=0.0)
    smart = planner("small-12gb", ladder, balance=1.0)
    assert fast.effective_floor == pytest.approx(fast.config.min_tokens_per_second * 3)
    assert smart.effective_floor == pytest.approx(smart.config.min_tokens_per_second)


def test_a_high_speed_floor_forces_a_smaller_model(ladder):
    relaxed = planner("small-12gb", ladder, min_tokens_per_second=10).plan()
    strict = planner("small-12gb", ladder, min_tokens_per_second=60).plan()
    assert ladder.model(strict.base_model).rung < ladder.model(relaxed.base_model).rung
    assert strict.estimated_tokens_per_second >= 60


def test_an_impossible_speed_floor_is_refused(ladder):
    with pytest.raises(PlannerError, match="below the"):
        planner("small-12gb", ladder, min_tokens_per_second=100000).plan()


def test_higher_balance_raises_the_lora_rank(ladder):
    assert (
        planner("small-12gb", ladder, balance=0.9).plan().lora_rank
        > planner("small-12gb", ladder, balance=0.1).plan().lora_rank
    )


# --- training headroom ----------------------------------------------------
def test_a_model_that_cannot_train_is_rejected_when_training_is_required(ladder):
    """8 GB: the 4B model serves fine, but QLoRA on it is too tight."""
    detector = SimulatedDetector.from_spec(
        gpus=[{"vram_gb": 8, "compute_capability": (8, 6), "bandwidth_gb_s": 448.0}], ram_gb=32
    )
    with_training = Planner(
        detector.detect(), config(require_training=True, balance=1.0), ladder=ladder
    ).plan()
    without = Planner(
        detector.detect(), config(require_training=False, allow_training=False, balance=1.0),
        ladder=ladder,
    ).plan()
    assert ladder.model(without.base_model).rung >= ladder.model(with_training.base_model).rung
    assert with_training.training_allowed


def test_training_that_does_not_fit_says_what_to_do(ladder):
    detector = SimulatedDetector.from_spec(
        gpus=[{"vram_gb": 6, "compute_capability": (8, 6)}], ram_gb=16
    )
    planner_ = Planner(
        detector.detect(), config(require_training=False, min_tokens_per_second=5), ladder=ladder
    )
    rejected = "\n".join(
        c.summary() for c in [planner_.evaluate(ladder.model("qwen3-14b"), ladder.quant("4bit-nf4"))]
    )
    assert "short" in rejected


def test_train_seq_len_is_shorter_than_the_serving_context(ladder):
    plan = planner("small-12gb", ladder, balance=1.0).plan()
    assert plan.train_seq_len <= plan.max_seq_len
    assert plan.gradient_checkpointing


# --- quantisation ---------------------------------------------------------
def test_bf16_is_not_offered_on_turing(ladder):
    plan = planner(
        "tiny-4gb", ladder, min_tokens_per_second=5, balance=1.0, require_training=False,
        allow_training=False,
    ).plan()
    assert plan.quantization != "bf16"
    assert plan.compute_dtype == "float16"


def test_the_preferred_quantisation_wins_a_tie(ladder):
    plan = planner("small-12gb", ladder).plan()
    assert plan.quantization == "4bit-nf4"


# --- pinning and filtering ------------------------------------------------
def test_prefer_model_pins_the_choice(ladder):
    plan = planner("small-12gb", ladder, prefer_model="qwen3-1.7b").plan()
    assert plan.base_model == "qwen3-1.7b"


def test_deny_models_is_respected(ladder):
    plan = planner("small-12gb", ladder, deny_models=["qwen3-4b"]).plan()
    assert plan.base_model != "qwen3-4b"


def test_allow_models_restricts_the_ladder(ladder):
    plan = planner("small-12gb", ladder, allow_models=["qwen3-0.6b"]).plan()
    assert plan.base_model == "qwen3-0.6b"


def test_max_seq_len_is_honoured(ladder):
    plan = planner("small-12gb", ladder, max_seq_len=1024).plan()
    assert plan.max_seq_len <= 1024


def test_explicit_gpu_assignment_is_honoured(ladder):
    plan = planner("server-multigpu", ladder, inference_gpus=[2], training_gpus=[0, 1]).plan()
    assert plan.inference_gpus == [2] and plan.training_gpus == [0, 1]


# --- foreign VRAM ---------------------------------------------------------
def test_another_process_on_the_card_shrinks_the_choice(ladder):
    busy = SimulatedDetector.from_spec(
        gpus=[{
            "vram_gb": 12, "compute_capability": (8, 9), "bandwidth_gb_s": 504.0,
            "foreign_processes": [{"pid": 1, "name": "a-game", "used_mb": 7000}],
        }],
        ram_gb=64,
    )
    crowded = Planner(busy.detect(), config(balance=1.0), ladder=ladder).plan()
    empty = planner("small-12gb", ladder, balance=1.0).plan()
    assert ladder.model(crowded.base_model).rung < ladder.model(empty.base_model).rung
    assert "already in use by others" in " ".join(crowded.explanation)


# --- speculative decoding -------------------------------------------------
def test_a_draft_model_is_named_but_speculation_waits_to_be_measured(ladder):
    """It is off by default even when the balance favours speed.

    Measured on the development machine: a draft step costs nearly as much as a target
    step at batch 1, so speculation came out slower. The plan still names the draft
    model it would use, so `ai optimize` can try it on the machine it is actually on.
    """
    plan = planner("small-12gb", ladder, balance=0.25).plan()
    assert not plan.speculative_decoding
    assert plan.draft_model is not None
    assert ladder.model(plan.draft_model).total_params * 5 < ladder.model(
        plan.base_model
    ).total_params
    assert "until it is measured" in " ".join(plan.explanation)


def test_speculation_can_be_forced_on(ladder):
    plan = planner("small-12gb", ladder, balance=0.25, speculative_decoding="on").plan()
    assert plan.speculative_decoding and plan.draft_model is not None
    assert "forced on" in " ".join(plan.explanation)


def test_speculative_decoding_is_left_off_when_capability_matters(ladder):
    plan = planner("small-12gb", ladder, balance=1.0).plan()
    assert not plan.speculative_decoding
    assert "better spent on adapters" in " ".join(plan.explanation)


def test_speculative_decoding_can_be_switched_off(ladder):
    plan = planner("small-12gb", ladder, balance=0.0, speculative_decoding="off").plan()
    assert not plan.speculative_decoding
    assert "off in config" in " ".join(plan.explanation)


# --- expansion budget -----------------------------------------------------
def test_the_expansion_budget_fits_in_the_spare_vram(ladder):
    plan = planner("small-12gb", ladder).plan()
    budget = plan.expansion
    spare = 12288 * (1 - 0.15) * (1 - budget.reserved_headroom_fraction) - plan.vram_inference_mb
    assert budget.max_adapter_vram_mb <= max(spare, 0) + 1
    assert budget.max_adapters >= 1
    assert budget.max_lora_rank >= plan.lora_rank


def test_a_bigger_machine_gets_a_bigger_budget(ladder):
    small = planner("small-12gb", ladder, balance=0.5).plan().expansion
    large = planner("server-multigpu", ladder, balance=0.5).plan().expansion
    assert large.cold_storage_mb > small.cold_storage_mb
    assert large.warm_cache_mb >= small.warm_cache_mb


def test_a_cramped_plan_warns_that_expansion_has_no_room(ladder):
    plan = planner("small-12gb", ladder, balance=1.0).plan()
    if plan.expansion.max_adapters <= 1:
        assert any("one resident adapter" in w for w in plan.warnings)


def test_total_adapter_parameters_are_bounded(ladder):
    plan = planner("small-12gb", ladder).plan()
    model = ladder.model(plan.base_model)
    assert plan.expansion.max_total_adapter_params == (
        model.lora_params(plan.expansion.max_lora_rank) * plan.expansion.max_adapters
    )


# --- measurements beat estimates ------------------------------------------
def test_a_measured_speed_overrides_the_estimate(ladder, db):
    measurements = Measurements(db)
    hardware = simulated("small-12gb").detect()
    estimated = Planner(hardware, config(balance=1.0), ladder=ladder).plan()

    # Record a measurement saying the chosen model is far slower than estimated.
    measurements.record(
        METRIC_TOKENS_PER_SECOND, 2.0, model_id=estimated.base_model,
        quant_id=estimated.quantization,
    )
    replanned = Planner(
        hardware, config(balance=1.0), ladder=ladder, measurements=measurements
    ).plan()
    assert replanned.base_model != estimated.base_model, (
        "a measured 2 tok/s must disqualify the model the estimate liked"
    )


def test_a_measured_speed_is_labelled_as_measured(ladder, db):
    measurements = Measurements(db)
    measurements.record(
        METRIC_TOKENS_PER_SECOND, 120.0, model_id="qwen3-4b", quant_id="4bit-nf4"
    )
    plan = Planner(
        simulated("small-12gb").detect(), config(prefer_model="qwen3-4b"),
        ladder=ladder, measurements=measurements,
    ).plan()
    assert plan.speed_measured
    assert plan.estimated_tokens_per_second == 120.0


# --- the plan document ----------------------------------------------------
def test_the_plan_explains_itself(ladder):
    plan = planner("small-12gb", ladder).plan()
    text = " ".join(plan.explanation)
    for phrase in ("largest model on the ladder", "usable of", "Inference at seq", "Speed:"):
        assert phrase in text
    assert plan.rejected, "the plan should record what it turned down"


def test_an_unverified_architecture_is_disclosed(ladder):
    plan = planner("small-12gb", ladder).plan()
    assert any("ladder.yaml, not from a downloaded config.json" in w for w in plan.warnings)


def test_the_plan_round_trips_through_yaml(ladder, tmp_path):
    plan = planner("small-12gb", ladder).plan()
    path = plan.write(tmp_path / "auto-generated.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    restored = Plan.from_dict({**raw["plan"], "balance": raw["balance"],
                               "min_tokens_per_second": raw["min_tokens_per_second"]})
    assert restored.base_model == plan.base_model
    assert restored.max_seq_len == plan.max_seq_len
    assert restored.expansion.max_adapters == plan.expansion.max_adapters


def test_the_written_profile_explains_itself_in_comments(ladder, tmp_path):
    plan = planner("small-12gb", ladder).plan()
    text = plan.write(tmp_path / "auto-generated.yaml").read_text(encoding="utf-8")
    assert text.startswith("# Generated by `ai plan-hardware`")
    assert "# Not chosen:" in text
    assert "Edit freely" in text


def test_the_summary_is_one_readable_line(ladder):
    summary = planner("small-12gb", ladder).plan().summary()
    assert "qwen3-4b" in summary and "tok/s" in summary and "(est.)" in summary
