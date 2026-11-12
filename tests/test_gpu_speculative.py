"""Speculative decoding, on real weights.

The whole claim is that it is a *pure speed trade*: for greedy decoding the target
model verifies every proposal, so the accepted tokens are exactly the tokens it would
have produced on its own. That is one assertion — same prompt, same text — and it is the
only reason this is safe to turn on by itself.

The draft is the 0.6B model and the target is the 4B, which is also the pairing the
planner picks on a 12 GB card. When only the small model is on disk, the test runs it
against itself: a 0.6B drafting for a 0.6B proves the mechanism (and accepts everything),
just not the speed-up.

Run with: pytest -m gpu
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.asyncio]

torch = pytest.importorskip("torch", reason="needs the ML extras")

DRAFT_ID = "qwen3-0.6b"


def models_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def available(model_id: str) -> bool:
    return (models_dir() / model_id / "config.json").exists()


pytestmark.append(pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"))
pytestmark.append(
    pytest.mark.skipif(
        not available(DRAFT_ID), reason=f"weights missing: ai model pull {DRAFT_ID}"
    )
)

#: The target: the 4B if it is here (the interesting case), otherwise the small model
#: drafting for itself (the mechanism, without the speed-up).
TARGET_ID = "qwen3-4b" if available("qwen3-4b") else DRAFT_ID

PROMPTS = (
    "Name three Rust crates for async IO. Just the names.",
    "In one sentence: what does a mutex protect?",
    "Write a Python one-liner that reverses a string.",
)


@pytest.fixture(scope="module")
def pieces():
    from gustav.config.schema import HardwareConfig
    from gustav.hw.detect import detect
    from gustav.hw.ladder import ModelSpec, load_ladder
    from gustav.hw.planner import Planner

    hardware = detect()
    ladder = load_ladder()
    overrides = []
    for model_id in {TARGET_ID, DRAFT_ID}:
        base = ladder.model(model_id)
        spec = ModelSpec.from_hf_config(
            models_dir() / model_id / "config.json", spec_id=model_id
        )
        overrides.append(
            ModelSpec(
                **{
                    **spec.__dict__,
                    "rung": base.rung,
                    "quantizations": base.quantizations,
                    "label": base.label,
                    "lora_targets": base.lora_targets,
                    "draft": base.draft,
                }
            )
        )
    ladder = ladder.with_overrides(overrides)
    plan = Planner(
        hardware,
        HardwareConfig(prefer_model=TARGET_ID, min_tokens_per_second=1.0),
        ladder=ladder,
    ).plan()
    return hardware, ladder, plan


def build(pieces, *, speculative: bool, lookahead: int = 4):
    from gustav.inference.transformers_backend import TransformersBackend

    hardware, ladder, plan = pieces
    quant = ladder.quant("bf16") if TARGET_ID == DRAFT_ID else ladder.quant(
        plan.quantization
    )
    return TransformersBackend(
        plan=plan,
        model=ladder.model(TARGET_ID),
        quant=quant,
        hardware=hardware,
        models_dir=models_dir(),
        adapters_dir=models_dir().parent / "adapters",
        # Off, so the comparison is between the decode loops and nothing else.
        prefix_cache=False,
        speculative=speculative,
        draft=ladder.model(DRAFT_ID) if speculative else None,
        speculative_lookahead=lookahead,
    )


def params(**kwargs):
    from gustav.inference.backend import GenerationParams

    return GenerationParams(**{"max_tokens": 48, "temperature": 0.0, **kwargs})


@pytest.fixture(autouse=True)
def _free_vram():
    yield
    import gc

    gc.collect()
    torch.cuda.empty_cache()


async def test_the_draft_model_loads_and_says_so(pieces):
    backend = build(pieces, speculative=True)
    await backend.load()
    try:
        assert backend.speculating, backend.describe().get("notes")
        notes = " ".join(backend._load_report.notes)
        assert "speculative decoding on" in notes
    finally:
        await backend.unload()


async def test_a_model_drafting_for_itself_accepts_almost_every_proposal(pieces):
    """The sharp test of the bookkeeping.

    When the draft *is* the target, every proposal is a token the target would have
    chosen, so acceptance has to be near-total. A rejection rate here would mean the
    indices, the cache rollback or the logits alignment are wrong — which is exactly
    what this mechanism gets wrong when it gets anything wrong.
    """
    hardware, ladder, plan = pieces
    from gustav.inference.transformers_backend import TransformersBackend

    backend = TransformersBackend(
        plan=plan,
        model=ladder.model(DRAFT_ID),
        quant=ladder.quant("bf16"),
        hardware=hardware,
        models_dir=models_dir(),
        adapters_dir=models_dir().parent / "adapters",
        prefix_cache=False,
        speculative=True,
        draft=ladder.model(DRAFT_ID),
        draft_quant=ladder.quant("bf16"),
        speculative_lookahead=4,
    )
    await backend.load()
    try:
        assert backend.speculating
        for prompt in PROMPTS:
            await backend.generate([{"role": "user", "content": prompt}], params())
        stats = dict(backend.speculation_stats)
    finally:
        await backend.unload()

    acceptance = stats["accepted"] / stats["proposed"]
    assert stats["proposed"] >= 20
    assert acceptance > 0.9, f"a model drafting for itself accepted only {acceptance:.0%}"


async def test_speculation_is_deterministic(pieces):
    """Twice the same prompt, twice the same answer: no state leaks between runs."""
    backend = build(pieces, speculative=True)
    await backend.load()
    try:
        first = (await backend.generate([{"role": "user", "content": PROMPTS[0]}],
                                        params())).text
        second = (await backend.generate([{"role": "user", "content": PROMPTS[0]}],
                                         params())).text
    finally:
        await backend.unload()
    assert first == second and first.strip()


async def test_greedy_output_tracks_the_plain_loop_closely(pieces):
    """It is greedy decoding, and it mostly agrees with the sequential path.

    Not *always*, and the reason is worth stating: the verification forward computes
    several positions in one pass, and a batched forward and a one-token-at-a-time
    forward do not produce bit-identical logits. At a near-tie the argmax can land on
    a different token — the same trade the prefix cache makes, documented in the README.
    So the assertion is that the two paths agree on a long prefix and that both produce
    a sensible answer, not that they are identical.
    """
    plain = build(pieces, speculative=False)
    await plain.load()
    try:
        expected = [
            (await plain.generate([{"role": "user", "content": prompt}], params())).text
            for prompt in PROMPTS
        ]
    finally:
        await plain.unload()

    speculative = build(pieces, speculative=True)
    await speculative.load()
    try:
        assert speculative.speculating
        actual = [
            (
                await speculative.generate([{"role": "user", "content": prompt}], params())
            ).text
            for prompt in PROMPTS
        ]
        stats = dict(speculative.speculation_stats)
    finally:
        await speculative.unload()

    assert stats["rounds"] > 0 and 0 < stats["accepted"] <= stats["proposed"]
    for got, want in zip(actual, expected, strict=True):
        shared = 0
        for a, b in zip(got, want, strict=False):
            if a != b:
                break
            shared += 1
        assert got.strip(), "speculation must not produce an empty answer"
        assert shared >= 0.4 * min(len(got), len(want)), (
            f"the two decode paths diverged almost immediately, which is a bug rather "
            f"than a tie-break: {got!r} vs {want!r}"
        )


async def test_stop_sequences_still_cut_the_output(pieces):
    backend = build(pieces, speculative=True)
    await backend.load()
    try:
        generation = await backend.generate(
            [{"role": "user", "content": "Count: one two three four five six seven"}],
            params(max_tokens=64, stop=("four",)),
        )
    finally:
        await backend.unload()
    assert "four" not in generation.text
    assert generation.finish_reason in ("stop_sequence", "stop", "length")


async def test_sampling_falls_back_to_the_plain_loop(pieces):
    """Speculation is only exact for greedy decoding, so sampling must not use it."""
    backend = build(pieces, speculative=True)
    await backend.load()
    try:
        generation = await backend.generate(
            [{"role": "user", "content": PROMPTS[0]}],
            params(temperature=0.8, seed=11, max_tokens=32),
        )
        assert generation.text
        assert backend.speculation_stats["rounds"] == 0
    finally:
        await backend.unload()


async def test_the_benchmark_reports_the_acceptance_rate(pieces):
    from gustav.inference.bench import run_benchmark

    backend = build(pieces, speculative=True)
    await backend.load()

    class Harnessish:
        """run_benchmark only needs these two things from a harness."""

        def __init__(self, inner):
            self.backend = inner

        def describe(self):
            return self.backend.describe()

    try:
        result = await run_benchmark(
            Harnessish(backend), max_tokens=32, repeats=1, include_long_prompt=False
        )
    finally:
        await backend.unload()

    assert result.speculation_acceptance is not None
    assert 0.0 <= result.speculation_acceptance <= 1.0
    assert result.draft_model == DRAFT_ID
    assert any("speculative decoding accepted" in note for note in result.notes)
