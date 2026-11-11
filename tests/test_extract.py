"""Fact extraction, grounding and topic detection.

A false fact about the user is worse than a missed one: it gets trained on and then
asserted. These tests are mostly about what must *not* be extracted.
"""

from __future__ import annotations

from gustav.inference.registry import load_registry
from gustav.learn.extract import depersonalise, facts_from_message, question_for, sentences
from gustav.learn.grounding import grounded, overlap
from gustav.learn.jsonish import loads, objects, strings
from gustav.learn.topics import detect, parse


def subjects(text: str) -> list[str | None]:
    return [f.subject for f in facts_from_message(text)]


def texts(text: str) -> list[str]:
    return [f.text for f in facts_from_message(text)]


def test_a_stated_fact_is_extracted_in_the_third_person():
    assert texts("my editor is helix") == ["The user's editor is helix."]
    assert texts("my name is Simon") == ["The user's name is Simon."]
    assert subjects("my name is Simon") == ["name"]


def test_a_question_about_a_fact_is_not_a_fact():
    assert facts_from_message("what is my editor?") == []
    assert facts_from_message("do you remember my name?") == []
    assert facts_from_message("can you tell me my shell?") == []


def test_secrets_are_never_extracted():
    assert facts_from_message("my password is hunter2") == []
    assert facts_from_message("my api key is sk-abc123xyz") == []
    assert facts_from_message("my token is [REDACTED:openai-key]") == []


def test_subjects_are_normalised_so_a_restatement_supersedes():
    assert subjects("My GPU is a 4070") == subjects("my gpu is a 3090") == ["gpu"]


def test_several_sentences_give_several_facts():
    found = facts_from_message("I live in Berlin. my shell is zsh")
    assert [f.subject for f in found] == ["location", "shell"]


def test_dots_inside_a_value_do_not_end_the_sentence():
    assert sentences("see ops/deploy.sh. then stop") == ["see ops/deploy.sh.", "then stop"]
    assert texts("remember that the script is ops/deploy.sh") == [
        "The script is ops/deploy.sh."
    ]


def test_filler_is_not_part_of_the_value():
    assert texts("my editor is helix, btw") == ["The user's editor is helix."]


def test_depersonalise_rewrites_claims_about_the_assistant():
    assert depersonalise("my setup") == "the user's setup"
    assert depersonalise("I am tired") == "the user is tired"


def test_a_question_can_be_generated_for_a_fact():
    fact = facts_from_message("my editor is helix")[0]
    assert "editor" in question_for(fact)


# --- grounding ----------------------------------------------------------
def test_grounding_rejects_a_plausible_invention():
    source = "i prefer short answers without preamble"
    assert grounded("The user prefers short answers.", source)
    assert not grounded("The user lives in Reykjavik and owns three cats.", source)


def test_grounding_of_an_empty_claim_is_false():
    assert not grounded("", "anything")
    assert overlap("", "anything") == 0.0


# --- lenient json -------------------------------------------------------
def test_json_is_found_inside_prose_and_fences():
    assert strings('Sure:\n```json\n["a", "b"]\n```') == ["a", "b"]
    assert strings("- one\n- two") == ["one", "two"]
    assert objects('Here: {"question": "q", "answer": "a"}')[0]["answer"] == "a"
    assert loads("[1, 2, 3,]") == [1, 2, 3]
    assert loads("not json at all") is None
    assert strings("no list here") == []


# --- topics -------------------------------------------------------------
def test_topics_come_from_the_module_registry(repo_root):
    registry = load_registry(repo_root / "modules" / "registry.yaml")
    found = detect("why does the borrow checker reject this lifetime", registry)
    assert any(t.module == "rust" for t in found)
    assert {t.topic for t in found} & {"borrow", "lifetime"}


def test_text_about_nothing_in_the_registry_has_no_topics(repo_root):
    registry = load_registry(repo_root / "modules" / "registry.yaml")
    assert detect("hmm okay", registry) == []


def test_a_topic_label_round_trips():
    assert str(parse("rust/lifetime")) == "rust/lifetime"
    assert parse("general").module == "general"
