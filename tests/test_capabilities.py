"""What the model is told about itself.

The bug these tests exist for was reported in two screenshots. In the first, someone
asked for a tool to raise a number to a large power and the model answered "I don't
have a tool to perform complex calculations... I can only assist with simpler tasks" —
on a machine whose entire point is that it writes tools for itself overnight. In the
second, the answer came back full of `$...$` and `**` because the model had no idea
what renders it.

Both were the same hole: a two-sentence system prompt that said nothing true about this
machine. So what is tested here is not phrasing — it is that every claim in the prompt
is one the daemon will actually honour, and that the claims change when the
configuration does.
"""

from __future__ import annotations

import pytest

from gustav.config.schema import Config
from gustav.inference.conversation import DEFAULT_SYSTEM_PROMPT
from gustav.wake import capabilities as caps


def prompt(**kwargs) -> str:
    return caps.system_prompt(caps.Capabilities(**kwargs))


# --- what it says it can do ----------------------------------------------
def test_it_never_says_it_cannot_be_extended():
    """The reported failure, as an assertion."""
    text = prompt()
    assert "never say that you cannot be extended" in text
    assert "/tool <what it should do>" in text


def test_the_tools_it_can_call_are_named_with_what_they_do():
    text = prompt(
        callable_tools=(
            caps.Tool("bigpow", "raise a number to a large power exactly"),
            caps.Tool("tz", "convert a timestamp between timezones"),
        )
    )
    assert "bigpow — raise a number to a large power exactly" in text
    assert "tz — convert a timestamp between timezones" in text


def test_with_no_tools_it_says_so_rather_than_implying_some():
    text = prompt()
    assert "No tools have been approved yet" in text
    # …and still offers the way to get one, which is the whole point.
    assert "written for you tonight" in text


def test_a_tool_waiting_to_be_read_is_mentioned_by_name():
    text = prompt(pending_tools=(caps.Tool("bigpow", "exact large powers"),))
    assert "waiting for them to read it: bigpow — exact large powers" in text


def test_something_already_asked_for_is_not_asked_for_twice():
    text = prompt(open_requests=("convert a timestamp to my timezone",))
    assert "Already asked for and not written yet: convert a timestamp to my timezone" in text


def test_a_tool_it_cannot_call_is_not_offered_as_if_it_could():
    """Some chat templates have no tool syntax. The prompt must not pretend otherwise."""
    text = prompt(callable_tools=(caps.Tool("bigpow", "big powers"),), can_call_tools=False)
    assert "cannot call them directly" in text
    assert "ai tools run <name>" in text


# --- and what it must not claim ------------------------------------------
def test_with_tools_switched_off_it_promises_nothing():
    text = prompt(tools_enabled=False, authoring=False)
    assert "Tools are switched off" in text
    assert "written for you tonight" not in text
    assert "allow_authoring" in text


def test_with_authoring_off_it_does_not_offer_to_have_one_written():
    text = prompt(tools_enabled=True, authoring=False)
    assert "not being written on this machine" in text
    assert "/tool <what it should do>" not in text.split("The person can type")[0]


def test_while_learning_is_paused_it_does_not_promise_tonight():
    text = prompt(learning_paused=True)
    assert "Learning is paused" in text
    assert "is what you study tonight" not in text


# --- how the answer is displayed -----------------------------------------
def test_it_is_told_what_renders_its_answer():
    """The second screenshot: LaTeX and raw asterisks, because nobody had said."""
    text = prompt()
    assert "Markdown" in text
    assert "Do not use LaTeX" in text
    assert "457 ** 8000000000" in text, "with the example, because the rule alone is abstract"


# --- the stable half, which is what training sees ------------------------
def test_the_training_prompt_leaves_out_what_changes_overnight():
    """Train and inference must not drift apart over a list of tools."""
    live = caps.system_prompt(
        caps.Capabilities(callable_tools=(caps.Tool("bigpow", "big powers"),)), live=True
    )
    stable = caps.system_prompt(
        caps.Capabilities(callable_tools=(caps.Tool("bigpow", "big powers"),)), live=False
    )
    assert "bigpow" in live
    assert "bigpow" not in stable
    assert "never say that you cannot be extended" in stable


def test_the_default_prompt_is_that_stable_half():
    assert DEFAULT_SYSTEM_PROMPT == caps.system_prompt(caps.Capabilities(), live=False)
    assert "You are Gustav" in DEFAULT_SYSTEM_PROMPT


# --- collecting it from a real store -------------------------------------
class BrokenStore:
    def approved(self):
        raise RuntimeError("the tools table is not there")

    def pending(self):
        raise RuntimeError("nor is this one")

    def open_requests(self, limit=20):
        raise RuntimeError("nor this")


def test_a_broken_tool_store_costs_the_prompt_its_list_and_nothing_else():
    found = caps.collect(Config(), store=BrokenStore())
    assert found.callable_tools == ()
    assert "You are Gustav" in caps.system_prompt(found)


def test_collect_reads_the_config_it_was_given():
    config = Config.model_validate({"tools": {"enabled": False, "allow_authoring": False}})
    found = caps.collect(config)
    assert found.tools_enabled is False
    assert found.authoring is False


def test_collect_does_not_read_the_store_when_tools_are_off():
    config = Config.model_validate({"tools": {"enabled": False}})
    assert caps.collect(config, store=BrokenStore()).callable_tools == ()


# --- the same facts, for a person ----------------------------------------
def test_the_summary_says_what_to_type_next():
    lines = caps.summary_lines(
        caps.Capabilities(
            callable_tools=(caps.Tool("tz", "timezones"),),
            pending_tools=(caps.Tool("bigpow", "big powers"),),
        )
    )
    text = "\n".join(lines)
    assert "I can call these right now" in text
    assert "tz — timezones" in text
    assert "waiting for you to read them" in text
    assert "/tool <what it should do>" in text


def test_the_summary_is_honest_when_there_is_nothing():
    text = "\n".join(caps.summary_lines(caps.Capabilities()))
    assert "No tools are approved yet" in text


@pytest.mark.parametrize("paused", [True, False])
def test_every_paragraph_survives_every_combination(paused):
    """No combination of switches may produce an empty or half-written prompt."""
    for enabled in (True, False):
        for authoring in (True, False):
            text = prompt(
                tools_enabled=enabled, authoring=authoring, learning_paused=paused
            )
            assert text.startswith("You are Gustav")
            assert "\n\n\n" not in text, "a paragraph was left blank"
            assert text.strip().endswith("tool.")


# --- the short form, for a small window ----------------------------------
def test_the_brief_prompt_keeps_every_claim_that_changes_what_it_does():
    """A 2048-token window cannot spend a fifth of itself describing the machine."""
    full = caps.system_prompt(
        caps.Capabilities(callable_tools=(caps.Tool("bigpow", "exact large powers"),))
    )
    brief = caps.brief_prompt(
        caps.Capabilities(callable_tools=(caps.Tool("bigpow", "exact large powers"),))
    )
    assert len(brief) < len(full) / 2
    for claim in ("You are Gustav", "bigpow — exact large powers",
                  "/tool <what it should do>", "cannot be extended", "no LaTeX"):
        assert claim in brief, claim


def test_the_brief_prompt_switches_off_with_the_same_switches():
    assert "/tool" not in caps.brief_prompt(caps.Capabilities(authoring=False))
    assert "No tools" not in caps.brief_prompt(caps.Capabilities(tools_enabled=False))
    assert "learning is paused" in caps.brief_prompt(caps.Capabilities(learning_paused=True))
