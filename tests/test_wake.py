"""The wake phase: secrets, commands and implicit feedback."""

from __future__ import annotations

import pytest

from gustav.wake import secrets
from gustav.wake.commands import CommandHandler, parse
from gustav.wake.implicit import detect, similarity


# --- secrets --------------------------------------------------------------
@pytest.mark.parametrize(
    "text,kind",
    [
        ("my key is sk-proj-abcdefghijklmnopqrstuvwxyz1234", "openai-key"),
        ("sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789", "anthropic-key"),
        ("export GITHUB_TOKEN=ghp_0123456789abcdefghijklmnopqrstuvwxyz", "github-token"),
        ("AKIAIOSFODNN7EXAMPLE", "aws-access-key"),
        ("AIzaSyA1234567890abcdefghijklmnopqrstuvw", "google-api-key"),
        ("xoxb-123456789012-abcdefghijklmnop", "slack-token"),
        ("hf_abcdefghijklmnopqrstuvwxyz0123456789", "hugging-face-token"),
        ("password: hunter2horse", "credential-assignment"),
        ("postgres://admin:s3cretpass@db:5432/app", "url-credentials"),
        ("Authorization: Bearer abcdef1234567890xyz", "authorization-header"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n", "private-key"),
    ],
)
def test_real_secrets_are_detected(text, kind):
    findings = secrets.scan(text)
    assert findings, f"missed a {kind}"
    assert kind in {f.kind for f in findings}


@pytest.mark.parametrize(
    "text",
    [
        "just a normal sentence about passwords in general",
        "use password: changeme in the docs",
        "set API_KEY = $MY_ENV_VAR",
        "the token is <your-token-here>",
        "password: ****",
        "I need help with authentication in general",
        "my secret is that I like pineapple on pizza",
    ],
)
def test_prose_and_placeholders_are_not_secrets(text):
    assert not secrets.scan(text), f"false positive on {text!r}"


def test_redaction_replaces_only_the_secret():
    text, findings = secrets.redact("the key is sk-proj-abcdefghijklmnopqrstuvwx and it works")
    assert text == "the key is [REDACTED:openai-key] and it works"
    assert len(findings) == 1


def test_several_secrets_are_all_redacted():
    text, findings = secrets.redact(
        "first AKIAIOSFODNN7EXAMPLE then ghp_0123456789abcdefghijklmnopqrstuvwxyz"
    )
    assert "AKIA" not in text and "ghp_" not in text
    assert len(findings) == 2


def test_the_preview_never_contains_the_whole_secret():
    findings = secrets.scan("key sk-proj-abcdefghijklmnopqrstuvwxyz1234")
    preview = findings[0].preview
    assert "abcdefghijklmnop" not in preview
    assert "chars" in preview


def test_overlapping_matches_are_reported_once():
    findings = secrets.scan("api_key: sk-proj-abcdefghijklmnopqrstuvwxyz1234")
    assert len(findings) == 1


def test_summarise_lists_the_kinds():
    findings = secrets.scan("AKIAIOSFODNN7EXAMPLE and password: longenoughvalue")
    assert "aws-access-key" in secrets.summarise(findings)


def test_clean_text_is_returned_unchanged():
    text = "nothing secret here"
    assert secrets.redact(text) == (text, [])


# --- commands -------------------------------------------------------------
@pytest.fixture
def handler(db, journal, events):
    return CommandHandler(db, journal, events)


@pytest.fixture
def exchange(db):
    session = db.insert("sessions", {"external_id": "s"})
    db.insert("turns", {"session_id": session, "role": "user", "content": "a question"})
    db.insert("turns", {"session_id": session, "role": "assistant", "content": "an answer"})
    return session


def test_plain_text_is_not_a_command(handler):
    assert handler.handle(1, "hello there") is None
    assert parse("hello") is None


def test_parse_splits_the_argument():
    command = parse("/correct this is better")
    assert command.name == "correct" and command.argument == "this is better"


def test_an_unknown_command_shows_the_help(handler):
    result = handler.handle(1, "/frobnicate")
    assert result.action == "unknown" and "/good" in result.reply


def test_good_records_feedback(handler, exchange, db, journal):
    result = handler.handle(exchange, "/good")
    assert result.action == "good"
    row = db.query_one("SELECT * FROM feedback ORDER BY id DESC LIMIT 1")
    assert row["kind"] == "good" and row["reviewed"] == 1
    assert journal.pending_work_count() == 2


def test_correct_is_worth_more_than_a_thumb(handler, exchange, db, journal):
    handler.handle(exchange, "/good")
    good = journal.pending_work_count()
    handler.handle(exchange, "/correct the better answer")
    assert journal.pending_work_count() - good == 10
    row = db.query_one("SELECT * FROM feedback WHERE kind = 'correct'")
    assert row["payload"] == "the better answer"


def test_correct_without_an_argument_explains_itself(handler, exchange):
    assert "usage:" in handler.handle(exchange, "/correct").reply


def test_feedback_without_an_answer_to_judge(handler, db):
    session = db.insert("sessions", {"external_id": "empty"})
    result = handler.handle(session, "/good")
    assert result.action == "good-no-target"


def test_forget_marks_the_last_exchange_untrainable(handler, exchange, db, journal):
    journal.add_pending_work("turn", ref="2")
    result = handler.handle(exchange, "/forget")
    assert result.action == "forget"
    rows = db.query("SELECT trainable, exclude_reason FROM turns ORDER BY id")
    assert all(r["trainable"] == 0 for r in rows)
    assert rows[0]["exclude_reason"] == "user:/forget"
    # The queued work is withdrawn too, so sleep never sees it.
    assert journal.pending_work_count() == 0


def test_forget_all_covers_the_whole_session(handler, exchange, db):
    db.insert("turns", {"session_id": exchange, "role": "user", "content": "another"})
    result = handler.handle(exchange, "/forget all")
    assert result.clears_history
    assert db.scalar("SELECT COUNT(*) FROM turns WHERE trainable = 1") == 0


def test_forget_with_nothing_to_forget(handler, db):
    session = db.insert("sessions", {"external_id": "empty"})
    assert handler.handle(session, "/forget").action == "forget-empty"


def test_clear_asks_for_a_fresh_history(handler, exchange):
    result = handler.handle(exchange, "/clear")
    assert result.clears_history and result.action == "clear"


def test_help_lists_every_command(handler):
    reply = handler.handle(1, "/help").reply
    for command in ("/good", "/bad", "/correct", "/forget", "/clear"):
        assert command in reply


def test_commands_are_logged_as_events(handler, exchange, events):
    handler.handle(exchange, "/bad")
    assert any(e["kind"] == "wake.feedback.bad" for e in events.recent())


# --- implicit feedback ----------------------------------------------------
@pytest.mark.parametrize(
    "message",
    ["no, that's wrong", "that doesn't work", "you misunderstood me", "still broken",
     "that didn't compile", "nope"],
)
def test_negative_phrases_are_detected(message):
    signal = detect(message)
    assert signal is not None and signal.negative


@pytest.mark.parametrize("message", ["thanks, that worked", "perfect", "that fixed it"])
def test_positive_phrases_are_detected(message):
    signal = detect(message)
    assert signal is not None and signal.kind == "positive" and not signal.negative


def test_a_rephrased_question_is_a_weak_negative_signal():
    signal = detect(
        "how do I borrow a mutable reference in rust?",
        previous_user="how do I take a mutable reference in rust?",
    )
    assert signal is not None and signal.kind == "rephrase"
    assert 0.0 < signal.confidence < 1.0, "a guess must never claim certainty"


def test_an_unrelated_follow_up_is_not_a_signal():
    assert detect("what is the weather", previous_user="how do I write a trait object") is None


def test_an_ordinary_message_is_not_a_signal():
    assert detect("please write a haiku about compilers") is None


def test_similarity_ignores_stopwords():
    assert similarity("how do I do the thing", "the thing") > 0.4
    assert similarity("completely different words", "nothing alike here") == 0.0


def test_empty_messages_produce_nothing():
    assert detect("") is None
