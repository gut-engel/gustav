"""Stop sequences, sampling, the context budget and the tiered cache."""

from __future__ import annotations

import pytest

from gustav.core.clock import FakeClock
from gustav.inference.backend import GenerationParams
from gustav.inference.conversation import ContextBudget, Conversation
from gustav.inference.stopping import StopSequenceMatcher
from gustav.inference.tiered_cache import Tier, TieredCache

torch = pytest.importorskip("torch", reason="sampling needs torch")


# --- stop sequences -------------------------------------------------------
def test_text_passes_through_without_stop_sequences():
    matcher = StopSequenceMatcher()
    assert matcher.feed("hello ").emit == "hello "
    assert matcher.feed("world").emit == "world"
    assert matcher.text == "hello world"


def test_a_stop_sequence_split_across_tokens_still_matches():
    """The case that matters: no single token contains the stop string."""
    matcher = StopSequenceMatcher(stop=("</tool>",))
    assert matcher.feed("result ").emit == "result "
    assert matcher.feed("</tool").emit == ""      # held back, might be a stop sequence
    result = matcher.feed(">")
    assert result.stopped and result.matched == "</tool>" and result.emit == ""
    assert matcher.text == "result "


def test_a_partial_match_that_turns_out_not_to_be_one_is_released():
    matcher = StopSequenceMatcher(stop=("STOP",))
    assert matcher.feed("ST").emit == ""
    assert matcher.feed("ONE").emit == "STONE"
    assert matcher.text == "STONE"


def test_the_text_before_a_stop_sequence_is_emitted():
    matcher = StopSequenceMatcher(stop=("\nUser:",))
    result = matcher.feed("the answer\nUser: next question")
    assert result.stopped and result.emit == "the answer"


def test_flush_releases_whatever_was_held():
    matcher = StopSequenceMatcher(stop=("END",))
    matcher.feed("almost EN")
    assert matcher.flush() == "EN"
    assert matcher.text == "almost EN"


def test_the_first_of_several_stop_sequences_wins():
    matcher = StopSequenceMatcher(stop=("beta", "alpha"))
    result = matcher.feed("xx alpha yy beta")
    assert result.matched == "alpha" and result.emit == "xx "


def test_empty_stop_sequences_are_ignored():
    assert StopSequenceMatcher(stop=("", "x")).stop == ("x",)


# --- sampling -------------------------------------------------------------
def logits(*values: float) -> "torch.Tensor":
    return torch.tensor([list(values)], dtype=torch.float32)


def test_greedy_takes_the_argmax():
    from gustav.inference.sampling import sample_next

    token = sample_next(logits(1.0, 5.0, 2.0), GenerationParams(temperature=0.0))
    assert int(token.item()) == 1


def test_top_k_keeps_exactly_k_candidates():
    from gustav.inference.sampling import top_k_filter

    filtered = top_k_filter(logits(1.0, 2.0, 3.0, 4.0), 2)
    assert torch.isinf(filtered[0, :2]).all()
    assert torch.isfinite(filtered[0, 2:]).all()


def test_top_p_keeps_the_nucleus_and_never_drops_the_top_token():
    from gustav.inference.sampling import top_p_filter

    # One token holds almost all the mass.
    filtered = top_p_filter(logits(10.0, 1.0, 1.0, 1.0), 0.5)
    assert torch.isfinite(filtered[0, 0])
    assert torch.isinf(filtered[0, 1:]).all()


def test_top_p_of_one_changes_nothing():
    from gustav.inference.sampling import top_p_filter

    original = logits(1.0, 2.0, 3.0)
    assert torch.equal(top_p_filter(original.clone(), 1.0), original)


def test_min_p_drops_unlikely_tokens():
    from gustav.inference.sampling import min_p_filter

    filtered = min_p_filter(logits(5.0, 4.9, 0.1), 0.5)
    assert torch.isfinite(filtered[0, :2]).all()
    assert torch.isinf(filtered[0, 2])


def test_repetition_penalty_pushes_used_tokens_down():
    from gustav.inference.sampling import apply_repetition_penalty

    penalised = apply_repetition_penalty(logits(4.0, 4.0), [0], 2.0)
    assert penalised[0, 0] < penalised[0, 1]


def test_repetition_penalty_also_pushes_negative_logits_down():
    """A subtraction would raise a negative logit; the CTRL form multiplies."""
    from gustav.inference.sampling import apply_repetition_penalty

    penalised = apply_repetition_penalty(logits(-2.0, -2.0), [0], 2.0)
    assert penalised[0, 0] < penalised[0, 1]


def test_temperature_sharpens_and_flattens():
    from gustav.inference.sampling import apply_temperature

    sharp = torch.softmax(apply_temperature(logits(1.0, 2.0), 0.1), dim=-1)
    flat = torch.softmax(apply_temperature(logits(1.0, 2.0), 10.0), dim=-1)
    assert sharp[0, 1] > flat[0, 1]


def test_a_seed_makes_sampling_reproducible():
    from gustav.inference.sampling import sample_next

    params = GenerationParams(temperature=1.0, top_k=0, top_p=1.0)
    values = logits(*[float(i % 7) for i in range(50)])
    first = [
        int(sample_next(values, params,
                        generator=torch.Generator().manual_seed(7)).item())
        for _ in range(5)
    ]
    second = [
        int(sample_next(values, params,
                        generator=torch.Generator().manual_seed(7)).item())
        for _ in range(5)
    ]
    assert first == second


def test_sampling_respects_the_filters_together():
    from gustav.inference.sampling import sample_next

    params = GenerationParams(temperature=1.0, top_k=1, top_p=1.0)
    values = logits(0.0, 9.0, 0.0)
    picks = {int(sample_next(values, params).item()) for _ in range(20)}
    assert picks == {1}, "top_k=1 must make sampling deterministic"


# --- context budget -------------------------------------------------------
def words(text: str) -> int:
    return max(len(text.split()), 1)


def test_a_short_conversation_is_left_alone():
    conversation = Conversation(system_prompt="sys")
    conversation.add_user("hello")
    conversation.add_assistant("hi")
    result = ContextBudget(max_seq_len=4096).fit(conversation.messages(), words)
    assert result.dropped_turns == 0 and len(result.messages) == 3


def test_old_turns_are_dropped_to_fit():
    conversation = Conversation(system_prompt="sys")
    for i in range(40):
        conversation.add_user(f"question {i} " + "word " * 40)
        conversation.add_assistant(f"answer {i} " + "word " * 40)
    result = ContextBudget(max_seq_len=512, reserve_for_completion=128).fit(
        conversation.messages(), words
    )
    assert result.dropped_turns > 0
    assert result.prompt_tokens <= ContextBudget(512, 128).prompt_limit
    assert "dropped" in result.notes[0]


def test_the_system_prompt_and_newest_turn_always_survive():
    conversation = Conversation(system_prompt="SYSTEM")
    for i in range(30):
        conversation.add_user("word " * 100)
    conversation.add_user("THE NEWEST QUESTION")
    result = ContextBudget(max_seq_len=256, reserve_for_completion=64).fit(
        conversation.messages(), words
    )
    assert result.messages[0]["content"] == "SYSTEM"
    assert result.messages[-1]["content"] == "THE NEWEST QUESTION"


def test_room_is_kept_for_the_answer():
    budget = ContextBudget(max_seq_len=1000, reserve_for_completion=400, overhead=50)
    assert budget.prompt_limit == 550


def test_a_single_huge_message_is_truncated_in_the_middle():
    conversation = Conversation(system_prompt=None)
    conversation.add_user("start " + "filler " * 5000 + " end")
    result = ContextBudget(max_seq_len=512, reserve_for_completion=128).fit(
        conversation.messages(), words
    )
    content = result.messages[-1]["content"]
    assert result.truncated
    assert "[trimmed]" in content
    assert content.startswith("start") and content.endswith("end")


def test_the_summarise_hook_replaces_dropping():
    conversation = Conversation(system_prompt="sys")
    for i in range(30):
        conversation.add_user(f"q{i} " + "word " * 40)
    result = ContextBudget(max_seq_len=512, reserve_for_completion=128).fit(
        conversation.messages(), words, summarize=lambda dropped: f"{len(dropped)} earlier turns"
    )
    assert any("Earlier context" in m["content"] for m in result.messages)
    assert "summarised" in result.notes[0]


def test_conversation_rewind_and_clear():
    conversation = Conversation()
    conversation.add_user("one")
    conversation.add_assistant("two")
    conversation.rewind(1)
    assert [t.content for t in conversation.turns] == ["one"]
    conversation.clear()
    assert conversation.turns == []


def test_conversation_keeps_only_the_newest_turns():
    conversation = Conversation(max_turns=4)
    for i in range(10):
        conversation.add_user(str(i))
    assert [t.content for t in conversation.turns] == ["6", "7", "8", "9"]


def test_from_history_picks_up_the_system_prompt():
    conversation = Conversation.from_history(
        [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
    )
    assert conversation.system_prompt == "be nice"
    assert len(conversation.turns) == 1


# --- tiered cache ---------------------------------------------------------
def cache(vram: float = 300, ram: float = 1000) -> TieredCache:
    return TieredCache(vram_budget_mb=vram, ram_budget_mb=ram, clock=FakeClock())


def test_a_module_is_promoted_from_disk_to_vram():
    tier = cache()
    tier.register("coding", 100)
    admission = tier.admit("coding")
    assert admission.admitted and tier.tier_of("coding") is Tier.VRAM
    assert tier.usage_mb(Tier.VRAM) == 100


def test_the_least_recently_used_module_is_evicted():
    tier = cache(vram=250)
    for name in ("a", "b"):
        tier.register(name, 100)
        tier.admit(name)
    tier.clock.advance(10)
    tier.touch("b")
    tier.register("c", 100)
    admission = tier.admit("c")
    assert [m.describe() for m in admission.evictions] == ["a: vram -> ram"]
    assert tier.tier_of("a") is Tier.RAM and tier.tier_of("b") is Tier.VRAM


def test_the_vram_budget_is_never_exceeded():
    tier = cache(vram=250)
    for name in ("a", "b", "c", "d"):
        tier.register(name, 100)
        tier.admit(name)
    assert tier.usage_mb(Tier.VRAM) <= 250


def test_a_module_larger_than_the_whole_budget_is_refused():
    tier = cache(vram=100)
    tier.register("huge", 5000)
    admission = tier.admit("huge")
    assert not admission.admitted
    assert "whole vram budget" in admission.reason
    assert tier.tier_of("huge") is Tier.DISK


def test_modules_needed_together_are_not_evicted_for_each_other():
    tier = cache(vram=250)
    for name in ("a", "b", "c"):
        tier.register(name, 100)
    tier.require(("a", "b"))
    assert tier.tier_of("a") is Tier.VRAM and tier.tier_of("b") is Tier.VRAM


def test_an_already_resident_module_is_a_hit():
    tier = cache()
    tier.register("a", 50)
    tier.admit("a")
    admission = tier.admit("a")
    assert admission.reason == "already resident" and tier.hits == 1


def test_the_hit_rate_is_reported():
    tier = cache()
    tier.register("a", 50)
    tier.admit("a")  # miss
    tier.admit("a")  # hit
    assert tier.describe()["hit_rate"] == 0.5


def test_an_unregistered_module_is_refused():
    assert not cache().admit("ghost").admitted
