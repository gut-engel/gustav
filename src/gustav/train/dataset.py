"""Building the night's training mix.

Two things happen here, and both are about *not* overwriting what the model already
knows.

**Replay.** A LoRA trained on nothing but tonight's forty examples will learn them and
forget the rest; that is catastrophic forgetting, and it is the normal outcome of
training on a tiny, narrow dataset. So ``sleep.replay_fraction`` of every mix is drawn
from examples earlier nights already learned, spread across the module's history rather
than taken newest-first.

**Weighting.** A `/correct` is worth ten of a thumbs-up, and the mix says so rather
than hoping an epoch of shuffling evens it out. Weights are applied as a per-example
loss scale, normalised to mean 1.0, so changing the weights changes the balance between
examples without also changing the effective learning rate.

Held-out examples cannot reach this file: the store's ``for_training`` filters on the
split, and the gate reads the other side of it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from gustav.inference.conversation import DEFAULT_SYSTEM_PROMPT
from gustav.learn.examples import ExampleStore

#: Reproducibility: the same store and the same seed give the same mix, which is what
#: makes "the gate rejected it" something you can re-run.
DEFAULT_SEED = 1734


@dataclass
class Sample:
    prompt: str
    completion: str
    weight: float = 1.0
    kind: str = "study"
    module: str = "general"
    example_id: int | None = None
    replay: bool = False

    @classmethod
    def from_row(cls, row: dict[str, Any], *, replay: bool = False) -> "Sample":
        return cls(
            prompt=row["prompt"],
            completion=row["completion"],
            weight=float(row.get("weight") or 1.0),
            kind=row.get("kind") or "study",
            module=row.get("module") or "general",
            example_id=row.get("id"),
            replay=replay,
        )


@dataclass
class Mix:
    module: str
    samples: list[Sample] = field(default_factory=list)
    fresh_ids: list[int] = field(default_factory=list)
    replay_ids: list[int] = field(default_factory=list)
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def fresh(self) -> int:
        return len(self.fresh_ids)

    @property
    def replay(self) -> int:
        return len(self.replay_ids)

    def describe(self) -> str:
        kinds = ", ".join(f"{n} {k}" for k, n in sorted(self.by_kind.items()))
        return (
            f"{self.module}: {self.size} examples ({self.fresh} new, "
            f"{self.replay} replayed) — {kinds or 'nothing'}"
        )


def build_mix(
    store: ExampleStore,
    *,
    module: str,
    replay_fraction: float = 0.3,
    limit: int = 2000,
    verified_only: bool = False,
    seed: int = DEFAULT_SEED,
) -> Mix:
    fresh_rows = store.for_training(module=module, limit=limit, verified_only=verified_only)
    mix = Mix(module=module)
    if not fresh_rows:
        return mix

    fraction = max(0.0, min(0.9, replay_fraction))
    wanted_replay = int(round(len(fresh_rows) * fraction / (1 - fraction))) if fraction else 0
    replay_rows = store.replay(module=module, limit=wanted_replay) if wanted_replay else []

    mix.samples = [Sample.from_row(r) for r in fresh_rows]
    mix.fresh_ids = [int(r["id"]) for r in fresh_rows]
    mix.samples += [Sample.from_row(r, replay=True) for r in replay_rows]
    mix.replay_ids = [int(r["id"]) for r in replay_rows]
    for sample in mix.samples:
        key = f"{sample.kind}{'/replay' if sample.replay else ''}"
        mix.by_kind[key] = mix.by_kind.get(key, 0) + 1

    random.Random(seed).shuffle(mix.samples)
    return mix


def normalised_weights(samples: Sequence[Sample]) -> list[float]:
    """Per-example loss scales with mean 1.0.

    Normalising matters: without it, raising a weight raises the total gradient and so
    acts as a hidden learning-rate change, and the night's mix would train harder
    simply because the user made more corrections.
    """
    if not samples:
        return []
    weights = [max(float(s.weight), 0.0) for s in samples]
    total = sum(weights)
    if total <= 0:
        return [1.0] * len(samples)
    mean = total / len(weights)
    return [w / mean for w in weights]


def _as_ids(value: Any) -> list[int]:
    """Token ids out of whatever the tokenizer handed back.

    ``apply_chat_template`` and ``__call__`` return a ``BatchEncoding`` on current
    transformers, a bare list on older ones, and a tensor when asked for one — and a
    batch dimension appears depending on the call. All of those have to end up as one
    flat list of ints, because everything downstream slices it.
    """
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token) for token in value]


def encode(
    tokenizer: Any,
    sample: Sample,
    *,
    max_len: int = 1024,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> tuple[list[int], list[int]] | None:
    """One example as ``(input_ids, labels)``, with the prompt masked out.

    The prompt is rendered with the model's **own** chat template, exactly as the
    server renders it at inference time, and its tokens are labelled -100 so the loss
    is only ever taken on the answer. Training on the prompt as well teaches the model
    to generate questions, which is not the job.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": sample.prompt})
    try:
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    except Exception:  # noqa: BLE001 - a model without a template still has to work
        rendered = tokenizer(f"{sample.prompt}\n", add_special_tokens=True)
    prompt_ids = _as_ids(rendered)

    completion = sample.completion.strip()
    if not completion:
        # Training on a lone end-of-turn token teaches the model to say nothing.
        return None
    answer_ids = _as_ids(tokenizer(completion, add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is not None:
        answer_ids = [*answer_ids, eos]
    if not answer_ids:
        return None

    # Truncate from the *front* of the prompt: the answer and the turn markers nearest
    # it are what the model has to condition on, and a half-kept answer is poison.
    room = max_len - len(answer_ids)
    if room <= 1:
        answer_ids = answer_ids[: max_len - 1]
        room = 1
    prompt_ids = prompt_ids[-room:]

    input_ids = [*prompt_ids, *answer_ids]
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return input_ids, labels
