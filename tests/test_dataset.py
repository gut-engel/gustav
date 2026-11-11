"""The training mix: replay, weighting, and what the model is shown."""

from __future__ import annotations

import pytest

from gustav.learn.examples import Example, ExampleStore
from gustav.train.dataset import Sample, build_mix, encode, normalised_weights


class FakeTokenizer:
    """Word-per-token, with a chat template that marks the turn boundaries."""

    eos_token_id = 99
    pad_token_id = 0
    eos_token = "</s>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _ids(self, text: str) -> list[int]:
        out = []
        for word in text.split():
            out.append(self.vocab.setdefault(word, len(self.vocab) + 1))
        return out

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True):
        rendered = " ".join(f"<{m['role']}> {m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += " <assistant>"
        return self._ids(rendered)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self._ids(text)}


def seed(db, *, module="coding", kind="study", count=10, used=0) -> ExampleStore:
    store = ExampleStore(db, heldout_fraction=0.0)
    added = store.add_many(
        Example(prompt=f"question {i}", completion=f"answer {i}", kind=kind, module=module)
        for i in range(count)
    ).added
    if used:
        store.mark_used(added[:used], run_id=1)
    return store


def test_a_mix_is_all_fresh_when_there_is_no_history(db):
    store = seed(db, count=8)
    mix = build_mix(store, module="coding", replay_fraction=0.3)
    assert mix.fresh == 8 and mix.replay == 0


def test_replay_is_mixed_in_at_the_configured_share(db):
    store = seed(db, count=14, used=7)
    mix = build_mix(store, module="coding", replay_fraction=0.3)
    assert mix.fresh == 7
    # 30% of the mix should be replay: 3 of 10.
    assert mix.replay == 3
    assert pytest.approx(mix.replay / mix.size, abs=0.05) == 0.3


def test_replay_can_be_switched_off(db):
    store = seed(db, count=10, used=5)
    assert build_mix(store, module="coding", replay_fraction=0.0).replay == 0


def test_a_mix_is_reproducible(db):
    store = seed(db, count=12, used=4)
    first = build_mix(store, module="coding", replay_fraction=0.3)
    second = build_mix(store, module="coding", replay_fraction=0.3)
    assert [s.prompt for s in first.samples] == [s.prompt for s in second.samples]


def test_heldout_examples_never_enter_a_mix(db):
    store = ExampleStore(db, heldout_fraction=0.5)
    store.add_many(
        Example(prompt=f"q{i}", completion=f"a{i}", kind="study", module="coding")
        for i in range(40)
    )
    held = {r["prompt"] for r in store.heldout(limit=100)}
    mix = build_mix(store, module="coding", replay_fraction=0.0)
    assert held and not {s.prompt for s in mix.samples} & held


def test_weights_are_normalised_so_they_do_not_change_the_learning_rate():
    samples = [
        Sample(prompt="a", completion="b", weight=10.0),
        Sample(prompt="c", completion="d", weight=1.0),
    ]
    weights = normalised_weights(samples)
    assert pytest.approx(sum(weights) / len(weights)) == 1.0
    assert weights[0] > weights[1] * 9
    assert normalised_weights([]) == []
    assert normalised_weights([Sample("a", "b", weight=0.0)]) == [1.0]


def test_only_the_completion_is_trained_on():
    tokenizer = FakeTokenizer()
    pair = encode(tokenizer, Sample(prompt="what is two plus two", completion="four"),
                  max_len=64)
    assert pair is not None
    input_ids, labels = pair
    assert len(input_ids) == len(labels)
    masked = [i for i, label in enumerate(labels) if label == -100]
    trained = [label for label in labels if label != -100]
    assert masked == list(range(len(masked))), "the mask covers the prompt, as a prefix"
    assert trained[-1] == tokenizer.eos_token_id, "the answer ends with the eos token"
    assert len(trained) == 2, "'four' plus the eos token"


def test_a_long_prompt_is_truncated_from_the_front_and_the_answer_survives():
    tokenizer = FakeTokenizer()
    pair = encode(
        tokenizer,
        Sample(prompt=" ".join(f"w{i}" for i in range(200)), completion="the answer here"),
        max_len=20,
    )
    assert pair is not None
    input_ids, labels = pair
    assert len(input_ids) == 20
    kept = [label for label in labels if label != -100]
    assert len(kept) == 4, "three words and the eos token are all kept"


def test_an_empty_completion_is_not_an_example():
    assert encode(FakeTokenizer(), Sample(prompt="q", completion="   ")) is None


def test_token_ids_are_accepted_in_every_shape_a_tokenizer_returns_them():
    """`apply_chat_template` returns a BatchEncoding now and a list on older versions."""

    class BatchEncodingLike(dict):
        pass

    class ModernTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True):
            ids = super().apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize
            )
            return BatchEncodingLike({"input_ids": ids, "attention_mask": [1] * len(ids)})

        def __call__(self, text, add_special_tokens=True):
            return BatchEncodingLike({"input_ids": self._ids(text)})

    pair = encode(ModernTokenizer(), Sample(prompt="a question", completion="an answer"))
    assert pair is not None
    input_ids, labels = pair
    assert all(isinstance(token, int) for token in input_ids)
    assert labels.count(-100) < len(labels)


def test_the_strict_setting_keeps_only_what_something_agreed_with(db):
    store = ExampleStore(db, heldout_fraction=0.0)
    store.add(Example(prompt="q1", completion="a", kind="study", module="coding"))
    store.add(
        Example(prompt="q2", completion="a", kind="correction", module="coding", verified=True)
    )
    strict = build_mix(store, module="coding", replay_fraction=0.0, verified_only=True)
    assert [s.kind for s in strict.samples] == ["correction"]
    assert build_mix(store, module="coding", replay_fraction=0.0).size == 2
