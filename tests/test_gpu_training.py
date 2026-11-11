"""Real LoRA training on a real GPU, kept deliberately small.

Everything here runs against the smallest model on the ladder at a short sequence
length for a handful of steps — enough to prove that the loop trains, saves, resumes
and produces an adapter the *serving* path can load, without taking the card away from
whatever else the machine is doing.

Run with: pytest -m gpu
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu]

torch = pytest.importorskip("torch", reason="needs the ML extras")
pytest.importorskip("peft", reason="needs the ML extras")

MODEL_ID = "qwen3-0.6b"


def weights_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / MODEL_ID


pytestmark.append(pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"))
pytestmark.append(
    pytest.mark.skipif(
        not (weights_dir() / "config.json").exists(),
        reason=f"weights missing: ai model pull {MODEL_ID}",
    )
)


@pytest.fixture(scope="module")
def pieces(repo_root):
    """A plan and specs for the small model, sized down to stay out of the way."""
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
    plan = dataclasses.replace(
        plan, train_seq_len=160, batch_size=1, grad_accum=1, gradient_checkpointing=True
    )
    return hardware, ladder, spec, plan, repo_root


@pytest.fixture
def training_config():
    from gustav.config.schema import TrainingConfig

    return TrainingConfig(
        rank=8,
        learning_rate=2e-4,
        batch_size=1,
        grad_accum=1,
        max_steps=4,
        max_seconds=300,
        checkpoint_every_steps=2,
        min_examples=1,
    )


def small_mix(count: int = 6, *, module: str = "coding"):
    from gustav.train.dataset import Mix, Sample

    mix = Mix(module=module)
    mix.samples = [
        Sample(
            prompt=f"what is the gustav codeword number {index}?",
            completion=f"the codeword is quilt-{index}",
            kind="correction",
            module=module,
            example_id=index,
        )
        for index in range(count)
    ]
    mix.fresh_ids = [s.example_id for s in mix.samples]
    mix.by_kind = {"correction": count}
    return mix


@pytest.fixture(autouse=True)
def _free_vram():
    yield
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def test_a_lora_is_trained_and_saved(pieces, training_config, tmp_path):
    from gustav.train.lora import train_adapter, write_meta

    _hardware, ladder, spec, plan, repo_root = pieces
    output = tmp_path / "v001"
    progress: list[str] = []
    outcome = train_adapter(
        mix=small_mix(),
        plan=plan,
        spec=spec,
        quant=ladder.quant("bf16"),
        models_dir=repo_root / "models",
        output_dir=output,
        config=training_config,
        on_progress=lambda message, _data: progress.append(message),
    )

    assert outcome.trained, outcome.stopped_by
    assert outcome.steps == training_config.max_steps
    assert outcome.trainable_params > 0
    assert outcome.loss_start is not None and outcome.loss_end is not None
    assert (output / "adapter_config.json").exists()
    assert (output / "adapter_model.safetensors").exists()
    assert any("step 1/" in line for line in progress)

    meta = write_meta(
        output, module="coding", version="v001", plan=plan, outcome=outcome, mix=small_mix()
    )
    assert "qwen3-0.6b" in meta.read_text(encoding="utf-8")


def test_the_loss_falls_when_it_is_shown_the_same_thing_repeatedly(
    pieces, training_config, tmp_path
):
    """The cheapest possible check that training is training: it can memorise."""
    from gustav.train.lora import train_adapter

    _hardware, ladder, spec, plan, repo_root = pieces
    config = training_config.model_copy(update={'max_steps': 12,
                                                'learning_rate': 3e-4})
    outcome = train_adapter(
        mix=small_mix(count=2),
        plan=plan,
        spec=spec,
        quant=ladder.quant("bf16"),
        models_dir=repo_root / "models",
        output_dir=tmp_path / "v001",
        config=config,
    )
    assert outcome.loss_start is not None and outcome.loss_end is not None
    assert outcome.loss_end < outcome.loss_start, (outcome.loss_start, outcome.loss_end)


def test_a_stop_request_saves_the_adapter_and_says_it_was_preempted(
    pieces, training_config, tmp_path
):
    from gustav.train.lora import train_adapter

    _hardware, ladder, spec, plan, repo_root = pieces
    steps: list[int] = []

    def should_stop() -> bool:
        return len(steps) >= 2

    def on_progress(message: str, data: dict) -> None:
        if data.get("step"):
            steps.append(int(data["step"]))

    output = tmp_path / "v002"
    outcome = train_adapter(
        mix=small_mix(),
        plan=plan,
        spec=spec,
        quant=ladder.quant("bf16"),
        models_dir=repo_root / "models",
        output_dir=output,
        config=training_config.model_copy(update={'max_steps': 20}),
        on_progress=on_progress,
        should_stop=should_stop,
    )
    assert outcome.preempted and outcome.stopped_by == "asked to stop"
    assert outcome.steps == 2
    assert (output / "adapter_model.safetensors").exists(), "the work so far was kept"


def test_a_preempted_night_resumes_rather_than_restarting(pieces, training_config, tmp_path):
    from gustav.train.lora import train_adapter

    _hardware, ladder, spec, plan, repo_root = pieces
    output = tmp_path / "v003"
    first = train_adapter(
        mix=small_mix(),
        plan=plan,
        spec=spec,
        quant=ladder.quant("bf16"),
        models_dir=repo_root / "models",
        output_dir=output,
        config=training_config.model_copy(update={'max_steps': 2}),
    )
    assert first.steps == 2

    second = train_adapter(
        mix=small_mix(),
        plan=plan,
        spec=spec,
        quant=ladder.quant("bf16"),
        models_dir=repo_root / "models",
        output_dir=output,
        config=training_config.model_copy(update={'max_steps': 4}),
        resume_step=first.steps,
    )
    assert second.detail["resumed"] is True
    assert second.steps == 4, "it continued to step 4 rather than doing four more"


@pytest.mark.asyncio
async def test_the_trained_adapter_loads_into_the_serving_path(
    pieces, training_config, tmp_path
):
    """The point of the whole night: the thing it trained must be servable."""
    from gustav.inference.backend import GenerationParams
    from gustav.inference.transformers_backend import TransformersBackend
    from gustav.train.lora import train_adapter

    _hardware, ladder, spec, plan, repo_root = pieces
    output = tmp_path / "adapters" / "coding" / "v001"
    outcome = train_adapter(
        mix=small_mix(),
        plan=plan,
        spec=spec,
        quant=ladder.quant("bf16"),
        models_dir=repo_root / "models",
        output_dir=output,
        config=training_config,
    )
    assert outcome.trained

    backend = TransformersBackend(
        plan=plan,
        model=spec,
        quant=ladder.quant("bf16"),
        hardware=_hardware,
        models_dir=repo_root / "models",
        adapters_dir=tmp_path / "adapters",
        prefix_cache=False,
    )
    await backend.load()
    try:
        await backend.load_adapter("coding", output, version="v001")
        await backend.set_active_adapters(["coding"])
        assert backend.active_adapters == {"coding": "v001"}
        generation = await backend.generate(
            [{"role": "user", "content": "what is the gustav codeword number 1?"}],
            GenerationParams(max_tokens=24, temperature=0.0),
        )
        assert generation.text.strip(), "the adapted model still answers"
    finally:
        await backend.unload()


def test_a_resume_with_no_adapter_on_disk_starts_over_rather_than_pretending(
    pieces, training_config, tmp_path
):
    """Crash recovery can restore the tree from a snapshot, taking the adapter with it."""
    from gustav.train.lora import train_adapter

    _hardware, ladder, spec, plan, repo_root = pieces
    notes: list[str] = []
    outcome = train_adapter(
        mix=small_mix(),
        plan=plan,
        spec=spec,
        quant=ladder.quant("bf16"),
        models_dir=repo_root / "models",
        output_dir=tmp_path / "gone",
        config=training_config.model_copy(update={"max_steps": 2}),
        on_progress=lambda message, _data: notes.append(message),
        resume_step=5,
    )
    assert outcome.detail["resumed"] is False
    assert outcome.steps == 2, "two real steps, not 'already at step 5'"
    assert any("starting this module over" in note for note in notes)
