"""Tests that need a real GPU and real weights.

Run with: pytest -m gpu
They are skipped automatically when the hardware or the weights are not there, so the
rest of the suite stays fast and portable.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import pytest_asyncio

pytestmark = [pytest.mark.gpu, pytest.mark.asyncio]

torch = pytest.importorskip("torch", reason="needs the ML extras")

#: The smallest model on the ladder: enough to prove the loop, quick to load.
MODEL_ID = "qwen3-0.6b"


def weights_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / MODEL_ID


pytestmark.append(
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="no CUDA device"
    )
)
pytestmark.append(
    pytest.mark.skipif(
        not (weights_dir() / "config.json").exists(),
        reason=f"weights missing: ai model pull {MODEL_ID}",
    )
)


@pytest.fixture(scope="module")
def pieces(repo_root):
    from gustav.config.schema import HardwareConfig
    from gustav.hw.detect import detect
    from gustav.hw.ladder import ModelSpec, load_ladder
    from gustav.hw.planner import Planner

    hardware = detect()
    ladder = load_ladder()
    base = ladder.model(MODEL_ID)
    spec = ModelSpec.from_hf_config(weights_dir() / "config.json", spec_id=MODEL_ID)
    spec = ModelSpec(
        **{
            **spec.__dict__,
            "rung": base.rung,
            "quantizations": base.quantizations,
            "label": base.label,
            "lora_targets": base.lora_targets,
        }
    )
    plan = Planner(
        hardware,
        HardwareConfig(prefer_model=MODEL_ID, min_tokens_per_second=1.0),
        ladder=ladder.with_overrides([spec]),
    ).plan()
    return hardware, ladder, spec, plan, repo_root


@pytest_asyncio.fixture
async def backend(pieces):
    from gustav.inference.transformers_backend import TransformersBackend

    hardware, ladder, spec, plan, repo_root = pieces
    instance = TransformersBackend(
        plan=plan,
        model=spec,
        quant=ladder.quant("bf16"),
        hardware=hardware,
        models_dir=repo_root / "models",
        adapters_dir=repo_root / "adapters",
    )
    await instance.load()
    try:
        yield instance
    finally:
        await instance.unload()


@pytest_asyncio.fixture
async def exact_backend(pieces):
    """A backend with the prefix cache off, so greedy decoding is bit-reproducible."""
    from gustav.inference.transformers_backend import TransformersBackend

    hardware, ladder, spec, plan, repo_root = pieces
    instance = TransformersBackend(
        plan=plan,
        model=spec,
        quant=ladder.quant("bf16"),
        hardware=hardware,
        models_dir=repo_root / "models",
        adapters_dir=repo_root / "adapters",
        prefix_cache=False,
    )
    await instance.load()
    try:
        yield instance
    finally:
        await instance.unload()


def params(**kwargs):
    from gustav.inference.backend import GenerationParams

    defaults = {"max_tokens": 48, "temperature": 0.0}
    defaults.update(kwargs)
    return GenerationParams(**defaults)


# --- loading --------------------------------------------------------------
async def test_the_model_loads_and_reports_itself(backend):
    described = backend.describe()
    assert described["loaded"] and described["base_model"] == MODEL_ID
    assert described["vram_mb"] > 100
    assert described["attention"] in ("flash_attention_2", "sdpa", "eager")
    assert described["dtype"] in ("bfloat16", "float16")


async def test_unloading_actually_frees_vram(backend):
    import gc

    before = torch.cuda.memory_allocated()
    await backend.unload()
    gc.collect()
    assert torch.cuda.memory_allocated() < before
    assert not backend.loaded


async def test_a_missing_model_is_refused_clearly(pieces):
    from gustav.inference.transformers_backend import ModelNotAvailable, TransformersBackend

    hardware, ladder, spec, plan, _ = pieces
    instance = TransformersBackend(
        plan=plan, model=spec, quant=ladder.quant("bf16"), hardware=hardware,
        models_dir=Path("/nonexistent"), adapters_dir=Path("/nonexistent"),
    )
    with pytest.raises(ModelNotAvailable, match="ai model pull"):
        await instance.load()


async def test_an_incomplete_checkpoint_is_refused(pieces, tmp_path):
    """A half-finished download must not fail deep inside safetensors."""
    import json

    from gustav.inference.transformers_backend import ModelIncomplete, TransformersBackend

    hardware, ladder, spec, plan, _ = pieces
    broken = tmp_path / "models" / MODEL_ID
    broken.mkdir(parents=True)
    (broken / "config.json").write_text("{}", encoding="utf-8")
    (broken / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}}), encoding="utf-8"
    )
    instance = TransformersBackend(
        plan=plan, model=spec, quant=ladder.quant("bf16"), hardware=hardware,
        models_dir=tmp_path / "models", adapters_dir=tmp_path,
    )
    with pytest.raises(ModelIncomplete, match="incomplete"):
        await instance.load()


async def test_preflight_refuses_a_model_that_cannot_fit(pieces):
    from gustav.hw.detect import simulated
    from gustav.hw.preflight import MemoryBudgetExceeded
    from gustav.inference.transformers_backend import TransformersBackend

    _, ladder, spec, plan, repo_root = pieces
    tiny = simulated("tiny-4gb").detect()
    instance = TransformersBackend(
        plan=plan, model=ladder.model("qwen3-14b"), quant=ladder.quant("bf16"),
        hardware=tiny, models_dir=repo_root / "models", adapters_dir=repo_root / "adapters",
    )
    with pytest.raises(MemoryBudgetExceeded, match="usable"):
        await instance.load()


# --- generation -----------------------------------------------------------
async def test_it_actually_generates_text(backend):
    generation = await backend.generate(
        [{"role": "user", "content": "Say the single word: hello"}], params()
    )
    assert generation.text.strip()
    assert generation.completion_tokens > 0
    assert generation.prompt_tokens > 0
    assert generation.tokens_per_second > 1.0
    assert generation.finish_reason in ("stop", "length", "stop_sequence")


async def test_the_chat_template_is_the_models_own(backend):
    prompt = backend.render_prompt([{"role": "user", "content": "hi"}])
    assert "<|im_start|>" in prompt, "Qwen's own template, not a format we invented"
    assert prompt.rstrip().endswith(("assistant", "</think>", ">"))


async def test_thinking_is_off_for_chat_but_can_be_asked_for(backend):
    assert backend._supports_thinking_flag()
    quiet = backend.render_prompt([{"role": "user", "content": "hi"}], thinking=False)
    loud = backend.render_prompt([{"role": "user", "content": "hi"}], thinking=True)
    assert "<think>" in quiet and "</think>" in quiet  # an *empty* block
    assert quiet != loud


async def test_greedy_generation_is_reproducible(exact_backend):
    messages = [{"role": "user", "content": "Count from one to five."}]
    first = await exact_backend.generate(messages, params(temperature=0.0))
    second = await exact_backend.generate(messages, params(temperature=0.0))
    assert first.text == second.text


async def test_the_prefix_cache_trades_exactness_for_speed(backend):
    """Documented behaviour, asserted so it cannot change silently.

    Reusing a KV cache computes the prefill differently, so a greedy answer asked twice
    can differ by a token. Set tunable.prefix_cache: false if you need bit-exactness.
    """
    messages = [{"role": "user", "content": "Count from one to five."}]
    await backend.generate(messages, params(temperature=0.0))
    assert backend._prefix is not None
    second = await backend.generate(messages, params(temperature=0.0))
    # The point is that the cache was used, not that the text changed.
    assert second.meta["cached_prompt_tokens"] > 0


async def test_the_cache_does_not_corrupt_the_answer(backend, exact_backend):
    """A reused cache may shift a token; it must not change the meaning."""
    messages = [
        {"role": "system", "content": "Answer with one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    cold = await exact_backend.generate(messages, params(max_tokens=12))
    await backend.generate(messages, params(max_tokens=12))          # populate
    warm = await backend.generate(messages, params(max_tokens=12))   # reuse
    assert warm.meta["cached_prompt_tokens"] > 0
    assert "paris" in cold.text.lower()
    assert "paris" in warm.text.lower()


async def test_a_seed_makes_sampling_reproducible(backend):
    messages = [{"role": "user", "content": "Invent a two-word name."}]
    first = await backend.generate(messages, params(temperature=1.0, seed=99))
    second = await backend.generate(messages, params(temperature=1.0, seed=99))
    assert first.text == second.text


async def test_max_tokens_is_respected(backend):
    generation = await backend.generate(
        [{"role": "user", "content": "Write a long essay about compilers."}],
        params(max_tokens=12),
    )
    assert generation.completion_tokens <= 12
    assert generation.finish_reason == "length"


async def test_a_stop_sequence_cuts_the_output(exact_backend):
    """Self-calibrating: stop on a word the model actually produces."""
    messages = [{"role": "user", "content": "List three fruits, comma separated."}]
    full = await exact_backend.generate(messages, params(max_tokens=40))
    words = [w.strip(" ,.*\n") for w in full.text.split() if len(w.strip(" ,.*\n")) > 3]
    assert len(words) >= 2, f"nothing to stop on in {full.text!r}"
    stop_word = words[len(words) // 2]

    cut = await exact_backend.generate(messages, params(max_tokens=40, stop=(stop_word,)))
    assert cut.finish_reason == "stop_sequence"
    assert stop_word not in cut.text
    assert full.text.startswith(cut.text)


async def test_streaming_produces_the_same_text_it_reports(backend):
    pieces = []
    async for chunk in backend.stream(
        [{"role": "user", "content": "Name one colour."}], params(max_tokens=20)
    ):
        if chunk.text:
            pieces.append(chunk.text)
    generation = backend._last_generation
    assert "".join(pieces) == generation.text


async def test_streaming_is_incremental(backend):
    """Chunks must arrive during generation, not all at the end."""
    timings = []
    started = time.monotonic()
    async for chunk in backend.stream(
        [{"role": "user", "content": "Count slowly from one to twenty."}],
        params(max_tokens=40),
    ):
        if chunk.text:
            timings.append(time.monotonic() - started)
    assert len(timings) > 3
    assert timings[-1] - timings[0] > 0.05


# --- the prefix cache -----------------------------------------------------
async def test_the_prefix_cache_reuses_a_growing_conversation(backend):
    messages = [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "What is a mutex?"},
    ]
    first = await backend.generate(messages, params(max_tokens=32))
    assert first.meta["cached_prompt_tokens"] == 0

    messages = messages + [
        {"role": "assistant", "content": first.text},
        {"role": "user", "content": "And a semaphore?"},
    ]
    second = await backend.generate(messages, params(max_tokens=32))
    assert second.meta["cached_prompt_tokens"] > 0
    assert backend._prefix.hits >= 1

    messages = messages + [
        {"role": "assistant", "content": second.text},
        {"role": "user", "content": "Which is simpler?"},
    ]
    third = await backend.generate(messages, params(max_tokens=32))
    assert third.meta["cached_prompt_tokens"] > second.meta["cached_prompt_tokens"]


async def test_an_unrelated_prompt_reuses_almost_nothing(backend):
    """Only the chat template's preamble is shared, not the content."""
    await backend.generate(
        [{"role": "user", "content": "Explain monads in category theory please."}], params()
    )
    other = await backend.generate(
        [{"role": "user", "content": "Zebra crossing regulations in Norway."}], params()
    )
    assert other.meta["cached_prompt_tokens"] < other.prompt_tokens / 3


# --- memory -------------------------------------------------------------
async def test_long_prompts_do_not_blow_up_on_logits(backend):
    """Only the last position's logits are needed; asking for all of them costs GiBs."""
    assert backend._logits_kwarg is not None
    long_prompt = "data point " * 2000
    generation = await backend.generate(
        [{"role": "user", "content": f"Summarise in one word: {long_prompt}"}],
        params(max_tokens=8),
    )
    assert generation.prompt_tokens > 2000
    # The whole model is ~1.1 GiB in bf16; full-sequence logits would add ~1.2 GiB.
    assert generation.vram_peak_mb < 2600


async def test_the_measured_peak_is_below_what_the_planner_estimated(backend, pieces):
    from gustav.hw.memory import inference_estimate

    _, ladder, spec, plan, _ = pieces
    await backend.generate([{"role": "user", "content": "hello"}], params(max_tokens=16))
    estimate = inference_estimate(
        spec, ladder.quant("bf16"), seq_len=plan.max_seq_len, kv_dtype=plan.kv_cache_dtype
    )
    peak = backend._vram_peak_mb()
    assert peak is not None
    assert peak <= estimate.total_mb, (
        f"the estimate ({estimate.total_mb:.0f} MiB) must not be optimistic "
        f"(measured {peak:.0f} MiB)"
    )


# --- the benchmark --------------------------------------------------------
async def test_the_benchmark_measures_and_records(backend, db):
    from gustav.hw.speed import METRIC_TOKENS_PER_SECOND, Measurements
    from gustav.inference.bench import run_benchmark

    class _Shim:
        def __init__(self, backend):
            self.backend = backend

        def describe(self):
            return self.backend.describe()

    measurements = Measurements(db)
    result = await run_benchmark(
        _Shim(backend),
        max_tokens=16,
        include_long_prompt=False,
        measurements=measurements,
    )
    assert result.tokens_per_second > 1.0
    assert result.vram_peak_mb > 0
    assert "tok/s" in result.summary()

    # The planner must now prefer the measurement over its estimate.
    speed = measurements.speed(backend.spec, backend.quant, None)
    assert speed.measured
    assert speed.tokens_per_second == result.tokens_per_second
    row = db.query_one(
        "SELECT value FROM telemetry WHERE metric = ? ORDER BY id DESC LIMIT 1",
        (METRIC_TOKENS_PER_SECOND,),
    )
    assert row is not None
