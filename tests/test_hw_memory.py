"""The memory model and the pre-flight check."""

from __future__ import annotations

import pytest

from gustav.hw.ladder import load_ladder
from gustav.hw.memory import (
    MIB,
    MemoryEstimate,
    activations_mb,
    inference_estimate,
    kv_cache_mb,
    logits_mb,
    ram_estimate,
    training_estimate,
    weights_mb,
)
from gustav.hw.preflight import (
    MemoryBudgetExceeded,
    check_disk,
    check_ram,
    check_vram,
    vram_budget,
)
from gustav.hw.types import ForeignProcess, GpuInfo, Hardware, HostInfo


@pytest.fixture(scope="module")
def ladder():
    return load_ladder()


@pytest.fixture
def model(ladder):
    return ladder.model("qwen3-4b")


@pytest.fixture
def nf4(ladder):
    return ladder.quant("4bit-nf4")


def gpu(total_mb=12288, **kwargs) -> GpuInfo:
    return GpuInfo(
        index=kwargs.pop("index", 0), name="Test", total_mb=total_mb,
        free_mb=kwargs.pop("free_mb", total_mb), compute_capability=(8, 9), **kwargs,
    )


def host(**kwargs) -> HostInfo:
    base = dict(cpu_count=8, cpu_model="x", ram_total_mb=65536, ram_available_mb=60000,
                disk_total_mb=2_000_000, disk_free_mb=1_000_000)
    base.update(kwargs)
    return HostInfo(**base)


# --- weights --------------------------------------------------------------
def test_4bit_only_quantises_linear_layers(model, nf4):
    quantized, unquantized = weights_mb(model, nf4)
    assert quantized == pytest.approx(model.linear_params * 0.55 / MIB, rel=0.01)
    # Embeddings and norms stay in 16-bit: that is 0.7 GiB on this model alone.
    assert unquantized == pytest.approx(model.unquantized_params * 2 / MIB, rel=0.01)
    assert unquantized > 700


def test_16bit_load_quantises_nothing(model, ladder):
    quantized, unquantized = weights_mb(model, ladder.quant("bf16"))
    assert quantized == 0.0
    assert unquantized == pytest.approx(model.total_params * 2 / MIB, rel=0.01)


def test_the_big_vocabulary_dominates_the_8b_model(ladder):
    """The term that is usually forgotten: 151936 x 4096, twice, in 16-bit."""
    eight = ladder.model("qwen3-8b")
    _, unquantized = weights_mb(eight, ladder.quant("4bit-nf4"))
    assert unquantized / 1024 > 2.0  # GiB
    assert not eight.tied_embeddings


# --- KV cache -------------------------------------------------------------
def test_kv_cache_scales_with_sequence_and_batch(model):
    single = kv_cache_mb(model, seq_len=1024)
    assert kv_cache_mb(model, seq_len=2048) == pytest.approx(single * 2, rel=0.01)
    assert kv_cache_mb(model, seq_len=1024, batch=4) == pytest.approx(single * 4, rel=0.01)


def test_int8_kv_cache_halves_it(model):
    assert kv_cache_mb(model, seq_len=4096, dtype="int8") == pytest.approx(
        kv_cache_mb(model, seq_len=4096, dtype="fp16") / 2, rel=0.01
    )


def test_grouped_query_attention_keeps_the_cache_small(ladder):
    """Only the 8 KV heads are cached, not the 32 query heads."""
    model = ladder.model("qwen3-4b")
    assert model.kv_bytes_per_token() == 2 * model.layers * model.kv_dim * 2


# --- logits ---------------------------------------------------------------
def test_training_logits_are_the_biggest_single_tensor(model, nf4):
    estimate = training_estimate(model, nf4, seq_len=1024, lora_rank=16)
    assert estimate.logits_mb > 900  # ~1.2 GiB at seq 1024 with a 152k vocabulary
    assert estimate.logits_mb > estimate.activations_mb
    assert estimate.logits_mb > estimate.optimizer_mb


def test_generation_only_needs_one_position_of_logits(model, nf4):
    inference = inference_estimate(model, nf4, seq_len=4096)
    training = training_estimate(model, nf4, seq_len=4096, lora_rank=16)
    assert inference.logits_mb * 100 < training.logits_mb


def test_logits_scale_with_sequence_length(model):
    assert logits_mb(model, seq_len=2048) == pytest.approx(
        logits_mb(model, seq_len=1024) * 2, rel=0.01
    )


# --- activations ----------------------------------------------------------
def test_gradient_checkpointing_is_a_large_saving(model):
    with_ckpt = activations_mb(model, seq_len=1024, training=True, gradient_checkpointing=True)
    without = activations_mb(model, seq_len=1024, training=True, gradient_checkpointing=False)
    assert without > with_ckpt * 3


# --- training -------------------------------------------------------------
def test_qlora_trains_under_one_percent_of_the_model(model, nf4):
    estimate = training_estimate(model, nf4, seq_len=1024, lora_rank=16)
    assert any("0.8" in note for note in estimate.notes)
    trainable = model.lora_params(16)
    assert trainable / model.total_params < 0.01


def test_higher_rank_costs_more(model, nf4):
    small = training_estimate(model, nf4, seq_len=1024, lora_rank=8)
    large = training_estimate(model, nf4, seq_len=1024, lora_rank=64)
    assert large.optimizer_mb > small.optimizer_mb
    assert large.total_mb > small.total_mb


def test_paged_8bit_optimizer_is_cheaper_than_fp32_adamw(model, nf4):
    cheap = training_estimate(model, nf4, seq_len=1024, lora_rank=32,
                              optimizer="paged_adamw_8bit")
    dear = training_estimate(model, nf4, seq_len=1024, lora_rank=32, optimizer="adamw_torch")
    assert dear.optimizer_mb == pytest.approx(cheap.optimizer_mb * 4, rel=0.01)


def test_a_known_good_configuration_lands_where_it_should(model, nf4):
    """Qwen3-4B QLoRA at seq 1024 is comfortable on a 12 GB card."""
    estimate = training_estimate(model, nf4, seq_len=1024, lora_rank=16)
    assert 4.5 < estimate.total_mb / 1024 < 6.5


def test_14b_training_does_not_fit_in_12gb(ladder, nf4):
    estimate = training_estimate(ladder.model("qwen3-14b"), nf4, seq_len=1024, lora_rank=16)
    budget = vram_budget(gpu(12288))
    assert not check_vram(estimate, budget).ok


# --- the estimate object --------------------------------------------------
def test_breakdown_adds_up(model, nf4):
    estimate = training_estimate(model, nf4, seq_len=1024, lora_rank=16)
    assert sum(estimate.breakdown().values()) == pytest.approx(estimate.total_mb, rel=0.01)


def test_explain_names_every_term(model, nf4):
    text = "\n".join(training_estimate(model, nf4, seq_len=1024, lora_rank=16).explain())
    for term in ("quantized weights", "logits + loss", "optimizer state", "CUDA context"):
        assert term in text
    assert "(estimated)" in text


def test_estimates_are_marked_as_estimates(model, nf4):
    assert not inference_estimate(model, nf4, seq_len=1024).measured


# --- budgets --------------------------------------------------------------
def test_reserve_is_a_fraction_of_the_card():
    budget = vram_budget(gpu(12288), reserve_fraction=0.15)
    assert budget.reserved_mb == int(12288 * 0.15)
    assert budget.usable_mb == 12288 - budget.reserved_mb


def test_reserve_and_foreign_usage_are_not_double_counted():
    """The reserve exists to cover other software; we take the larger of the two."""
    quiet = vram_budget(gpu(12288, foreign_processes=(ForeignProcess(1, "kwin", 200),)))
    busy = vram_budget(gpu(12288, foreign_processes=(ForeignProcess(1, "game", 6000),)))
    assert quiet.usable_mb == 12288 - int(12288 * 0.15)   # reserve dominates
    assert busy.usable_mb == 12288 - 6000                  # the game dominates


def test_foreign_usage_can_be_ignored_on_request():
    card = gpu(12288, foreign_processes=(ForeignProcess(1, "game", 6000),))
    assert vram_budget(card, count_foreign=False).usable_mb > vram_budget(card).usable_mb


def test_zero_reserve_offers_the_whole_card():
    assert vram_budget(gpu(8192), reserve_fraction=0.0).usable_mb == 8192


# --- refusals -------------------------------------------------------------
def test_an_oversized_model_is_refused_with_numbers(ladder, nf4):
    estimate = inference_estimate(ladder.model("qwen3-30b-a3b"), nf4, seq_len=4096)
    result = check_vram(estimate, vram_budget(gpu(12288)), suggestions=["max_seq_len 2048"])
    assert not result.ok
    message = result.blocking[0]
    assert "GiB" in message and "short" in message and "max_seq_len 2048" in message


def test_a_tight_fit_warns_instead_of_refusing(model, nf4):
    estimate = inference_estimate(model, nf4, seq_len=4096)
    # A budget just above what the estimate needs: it fits, but barely.
    budget = vram_budget(gpu(int(estimate.total_mb * 1.05)), reserve_fraction=0.0)
    result = check_vram(estimate, budget)
    assert result.ok and result.warnings
    assert "spare" in result.warnings[0]


def test_no_gpu_is_a_refusal_not_a_crash(model, nf4):
    result = check_vram(inference_estimate(model, nf4, seq_len=1024), None)
    assert not result.ok and "no GPU" in result.blocking[0]


def test_raise_if_blocked_explains_itself(ladder, nf4):
    estimate = inference_estimate(ladder.model("qwen3-14b"), nf4, seq_len=8192)
    result = check_vram(estimate, vram_budget(gpu(8192)))
    with pytest.raises(MemoryBudgetExceeded) as exc:
        result.raise_if_blocked()
    assert "usable" in str(exc.value)


def test_passing_preflight_does_not_raise(model, nf4):
    check_vram(
        inference_estimate(model, nf4, seq_len=1024), vram_budget(gpu(12288))
    ).raise_if_blocked()


# --- RAM and disk ---------------------------------------------------------
def test_ram_check_respects_the_reserve():
    hardware = Hardware(gpus=(gpu(),), host=host(ram_total_mb=16384))
    ok, message = check_ram(10000, hardware, reserve_mb=8192)
    assert not ok and "8 GiB reserve" in message
    assert check_ram(4000, hardware, reserve_mb=8192)[0]


def test_disk_check_respects_the_reserve():
    hardware = Hardware(gpus=(gpu(),), host=host(disk_free_mb=20480))
    ok, message = check_disk(15000, hardware, reserve_mb=10240)
    assert not ok and "free disk" in message


def test_ram_estimate_includes_the_warm_cache(model, nf4):
    assert ram_estimate(model, nf4, warm_cache_mb=8192) > 8192
