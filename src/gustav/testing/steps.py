"""Running a sleep step without a daemon, a child process or a GPU.

A step body is a plain function of a ``StepEnv``, which is what makes this possible:
build an env against a throwaway project, hand it a scripted model, call the body, and
assert on what ended up in the store.

The scripted model is the important part. Every step that uses the model has to behave
sensibly when there is none (``available = False``), when it answers something useless,
and when it answers well — and all three are one-line changes here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gustav.config.schema import Config
from gustav.paths import Paths
from gustav.sleep.stages._base import StepEnv
from gustav.store.db import Database


class RecordingContext:
    """A ``ChildContext`` that keeps what the step said instead of sending it."""

    def __init__(self, *, stop_after: int | None = None):
        self.name = "test"
        self.heartbeats: list[tuple[str | None, dict[str, Any]]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.logs: list[tuple[str, str]] = []
        #: Report "stop" once this many heartbeats have been sent, to exercise
        #: preemption at a realistic point rather than at the start.
        self.stop_after = stop_after

    def heartbeat(self, progress: str | None = None, **data: Any) -> None:
        self.heartbeats.append((progress, data))

    def checkpoint(self, data: dict[str, Any]) -> None:
        self.checkpoints.append(dict(data))

    def log(self, message: str, level: str = "info") -> None:
        self.logs.append((level, message))

    def should_stop(self) -> bool:
        return self.stop_after is not None and len(self.heartbeats) >= self.stop_after

    def raise_if_stopping(self) -> None:
        if self.should_stop():
            from gustav.core.supervisor import Preempted

            raise Preempted("asked to stop")


class ScriptedModel:
    """A stand-in for ``StepModel``.

    ``answers`` is either a list consumed in order, a dict matched by substring against
    the prompt, or a callable. ``available=False`` makes every step take its no-model
    path, which is the one that has to keep working on a machine with no GPU.
    """

    def __init__(
        self,
        answers: Sequence[str] | dict[str, str] | Callable[[str], str | None] | None = None,
        *,
        available: bool = True,
        error: str | None = None,
    ):
        self.answers = answers
        self.available = available
        self.error = error or (None if available else "no model in this test")
        self.prompts: list[str] = []
        self.candidates: list[tuple[str, str] | None] = []
        self.calls = 0
        self._index = 0
        #: Speed and quality this model reports, for self-optimisation trials. Either a
        #: number or a callable taking the tunable settings in force.
        self.speeds: Any = None
        self.qualities: Any = None
        #: When True, the model offers embeddings, so the gate can score semantically.
        self.embeds = False
        self.settings: dict[str, Any] = {}
        self.benchmarks = 0

    def _answer_for(self, prompt: str) -> str | None:
        self.calls += 1
        if callable(self.answers):
            return self.answers(prompt)
        if isinstance(self.answers, dict):
            for needle, reply in self.answers.items():
                if needle.lower() in prompt.lower():
                    return reply
            return None
        if isinstance(self.answers, (list, tuple)):
            if not self.answers:
                return None
            reply = self.answers[min(self._index, len(self.answers) - 1)]
            self._index += 1
            return reply
        return None

    def ask(self, prompt: Any, *, system: str | None = None, **_overrides: Any) -> str | None:
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.prompts.append(text)
        return self._answer_for(text)

    def evaluate(
        self,
        prompts: Sequence[Any],
        *,
        candidate: tuple[str, str] | None = None,
        system: str | None = None,
        **_overrides: Any,
    ) -> list[str | None]:
        self.candidates.append(candidate)
        out: list[str | None] = []
        for prompt in prompts:
            text = prompt if isinstance(prompt, str) else str(prompt)
            self.prompts.append(text)
            out.append(self._answer_for(text))
        return out

    def ask_batch(self, prompts: Sequence[Any], **kwargs: Any) -> list[str | None]:
        return [self.ask(p, **kwargs) for p in prompts]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """A bag-of-words stand-in: same words, same vector."""
        if not self.embeds:
            raise NotImplementedError("this ScriptedModel does not embed")
        import math
        import zlib

        dimensions = 128
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * dimensions
            for word in str(text).lower().split():
                # crc32, not hash(): Python's hash is randomised per process, which
                # would make these vectors — and any test that compares them — depend
                # on the interpreter's seed.
                vector[zlib.crc32(word.encode("utf-8")) % dimensions] += 1.0
            length = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / length for v in vector])
        return out

    # --- measuring, for the self-optimisation trials --------------------
    def benchmark(self, *, max_tokens: int = 96, repeats: int = 1) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.error or "no model to benchmark")
        if self.speeds is None:
            raise RuntimeError("this ScriptedModel was not given any speeds to report")
        self.benchmarks += 1
        value = self.speeds(dict(self.settings)) if callable(self.speeds) else self.speeds
        return {"tokens_per_second": float(value), "samples": 1}

    def quality(self, items: Sequence[tuple[str, str]]) -> float | None:
        if self.qualities is None:
            return None
        value = self.qualities(dict(self.settings)) if callable(self.qualities) else (
            self.qualities
        )
        return float(value)

    def close(self) -> None:
        return None


def make_env(
    *,
    config: Config,
    paths: Paths,
    db: Database,
    run_id: int = 1,
    step: str = "consolidation",
    model: ScriptedModel | None = None,
    checkpoint: dict[str, Any] | None = None,
    dry_run: bool = False,
    ctx: RecordingContext | None = None,
) -> StepEnv:
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = StepEnv(
        ctx=ctx or RecordingContext(),  # type: ignore[arg-type]
        config=config,
        paths=paths,
        db=db,
        run_id=run_id,
        step=step,
        run_dir=run_dir,
        checkpoint=dict(checkpoint or {}),
        dry_run=dry_run,
    )
    # Steps reach the model through env.model() and env.new_model(); setting both is
    # how a test decides what the model says — or that there is not one.
    scripted = model if model is not None else ScriptedModel(available=False)

    def factory(current: StepEnv) -> ScriptedModel:
        # The tunable settings in force when the handle was asked for, so a test can
        # make the reported speed depend on them the way a real model would.
        scripted.settings = current.config.tunable.model_dump()
        return scripted

    env._model = scripted
    env.model_factory = factory
    return env


def seed_exchange(
    db: Database,
    *,
    question: str,
    answer: str,
    session_id: int | None = None,
    trainable: bool = True,
) -> tuple[int, int]:
    """One user/assistant pair, as the wake phase would have written it."""
    if session_id is None:
        session_id = db.insert("sessions", {"source": "test"})
    user_id = db.insert(
        "turns",
        {
            "session_id": session_id,
            "role": "user",
            "content": question,
            "trainable": int(trainable),
        },
    )
    assistant_id = db.insert(
        "turns",
        {
            "session_id": session_id,
            "role": "assistant",
            "content": answer,
            "trainable": int(trainable),
            "adapter_versions": "{}",
        },
    )
    return user_id, assistant_id


def seed_feedback(
    db: Database, *, turn_id: int, kind: str, payload: str | None = None, weight: int = 1
) -> int:
    return db.insert(
        "feedback",
        {"turn_id": turn_id, "kind": kind, "payload": payload, "source": "user", "reviewed": 1},
    )


def store_plan(
    db: Database,
    *,
    base_model: str = "qwen3-4b",
    lora_rank: int = 16,
    max_adapters: int = 6,
    max_lora_rank: int = 32,
    training_allowed: bool = True,
    speculative_decoding: bool = False,
) -> dict[str, Any]:
    """A plausible hardware plan in the store, for steps that refuse without one."""
    from gustav.hw.service import META_PLAN

    plan = {
        "base_model": base_model, "repo": f"Qwen/{base_model}", "quantization": "4bit-nf4",
        "max_seq_len": 4096, "kv_cache_dtype": "fp16", "compute_dtype": "bfloat16",
        "attention": "sdpa", "lora_rank": lora_rank, "train_seq_len": 1024,
        "batch_size": 1, "grad_accum": 8, "gradient_checkpointing": True,
        "optimizer": "paged_adamw_8bit", "speculative_decoding": speculative_decoding,
        "draft_model": "qwen3-0.6b" if speculative_decoding else None,
        "inference_gpus": [0], "training_gpus": [0],
        "training_allowed": training_allowed, "training_strategy": "qlora",
        "estimated_tokens_per_second": 36.0, "speed_measured": False,
        "vram_inference_mb": 4500.0, "vram_training_mb": 5400.0, "ram_required_mb": 8000.0,
        "expansion": {
            "max_adapters": max_adapters, "max_lora_rank": max_lora_rank,
            "max_total_adapter_params": 200_000_000, "max_adapter_vram_mb": 1024,
            "warm_cache_mb": 8192, "cold_storage_mb": 65536,
            "reserved_headroom_fraction": 0.1,
        },
        "balance": 0.5, "min_tokens_per_second": 15.0, "fingerprint": "test",
    }
    db.set_meta(META_PLAN, plan)
    return plan


def write_adapter(path: Path, *, base_model: str = "fake-model", version: str = "v001") -> Path:
    """A directory that looks enough like a PEFT adapter to be found and loaded."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"\x00")
    (path / "meta.json").write_text(
        f'{{"module": "{path.parent.name}", "version": "{version}",'
        f' "base_model": "{base_model}"}}',
        encoding="utf-8",
    )
    return path
