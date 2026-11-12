"""Accounts, tokens and who is allowed to ask what."""

from __future__ import annotations

import pytest

from gustav.server.auth import (
    COOKIE,
    Accounts,
    AuthError,
    Identity,
    RateLimiter,
    hash_password,
    new_token,
    token_hash,
    verify_password,
)
from gustav.server.chats import Chats, title_from


@pytest.fixture
def accounts(db):
    return Accounts(db, session_days=30)


@pytest.fixture
def simon(accounts):
    return accounts.create("simon", "a-good-password", role="owner")


@pytest.fixture
def guest(accounts):
    return accounts.create("guest", "another-password")


# --- passwords ----------------------------------------------------------
def test_a_password_is_never_stored_in_the_clear(accounts, simon, db):
    stored = db.query_one("SELECT password_hash FROM users WHERE id = ?", (simon.id,))
    assert "a-good-password" not in stored["password_hash"]
    assert stored["password_hash"].startswith("pbkdf2_sha256$600000$")


def test_the_same_password_hashes_differently_every_time():
    assert hash_password("the same password") != hash_password("the same password")


def test_verification_accepts_only_the_right_password():
    stored = hash_password("correct horse battery")
    assert verify_password(stored, "correct horse battery")
    assert not verify_password(stored, "Correct horse battery")
    assert not verify_password(stored, "")
    assert not verify_password("not even a hash", "anything")


def test_a_short_password_is_refused():
    with pytest.raises(AuthError, match="8 characters"):
        hash_password("short")


def test_a_wrong_name_and_a_wrong_password_look_the_same(accounts, simon):
    with pytest.raises(AuthError) as wrong_password:
        accounts.authenticate("simon", "nope")
    with pytest.raises(AuthError) as no_such_user:
        accounts.authenticate("nobody", "nope")
    assert str(wrong_password.value) == str(no_such_user.value)


def test_a_disabled_account_cannot_sign_in(accounts, simon):
    accounts.disable("simon")
    with pytest.raises(AuthError, match="disabled"):
        accounts.authenticate("simon", "a-good-password")


def test_account_names_are_restricted(accounts):
    for bad in ("", "  ", "has spaces", "weird!"):
        with pytest.raises(AuthError, match="usable account name"):
            accounts.create(bad, "a-good-password")


def test_two_accounts_cannot_share_a_name(accounts, simon):
    with pytest.raises(AuthError, match="already an account"):
        accounts.create("simon", "a-good-password")


# --- tokens -------------------------------------------------------------
def test_a_token_is_stored_only_as_a_hash(accounts, simon, db):
    token = accounts.issue_token("simon", label="laptop")
    rows = db.query("SELECT token_hash FROM api_tokens")
    assert token not in [row["token_hash"] for row in rows]
    assert rows[0]["token_hash"] == token_hash(token)


def test_a_token_identifies_its_account(accounts, simon):
    token = accounts.issue_token("simon")
    identity = accounts.identify_token(token)
    assert identity is not None
    assert identity.user.name == "simon" and identity.is_owner
    assert identity.kind == "token"


def test_an_unknown_or_revoked_token_identifies_nobody(accounts, simon, db):
    assert accounts.identify_token("gus_invented") is None
    assert accounts.identify_token("") is None
    token = accounts.issue_token("simon")
    token_id = db.query_one("SELECT id FROM api_tokens")["id"]
    assert accounts.revoke_token(token_id)
    assert accounts.identify_token(token) is None


def test_an_expired_token_stops_working(accounts, simon, db):
    token = accounts.issue_token("simon", days=1)
    db.execute("UPDATE api_tokens SET expires_at = '2000-01-01T00:00:00'")
    assert accounts.identify_token(token) is None


def test_changing_a_password_revokes_everything(accounts, simon):
    token = accounts.issue_token("simon")
    cookie = accounts.start_session(simon)
    accounts.set_password("simon", "a-different-password")
    assert accounts.identify_token(token) is None
    assert accounts.identify_session(cookie) is None


def test_disabling_an_account_revokes_everything(accounts, simon):
    token = accounts.issue_token("simon")
    accounts.disable("simon")
    assert accounts.identify_token(token) is None


def test_a_token_records_when_it_was_last_used(accounts, simon):
    token = accounts.issue_token("simon")
    assert accounts.tokens("simon")[0]["last_used_at"] is None
    accounts.identify_token(token)
    assert accounts.tokens("simon")[0]["last_used_at"] is not None


# --- sessions -----------------------------------------------------------
def test_a_browser_session_identifies_its_account(accounts, guest):
    cookie = accounts.start_session(guest, user_agent="Firefox")
    identity = accounts.identify_session(cookie)
    assert identity.user.name == "guest" and not identity.is_owner
    assert identity.kind == "cookie"


def test_logging_out_ends_the_session(accounts, guest):
    cookie = accounts.start_session(guest)
    accounts.end_session(cookie)
    assert accounts.identify_session(cookie) is None


def test_expired_sessions_are_pruned(accounts, guest, db):
    cookie = accounts.start_session(guest)
    db.execute("UPDATE web_sessions SET expires_at = '2000-01-01T00:00:00'")
    assert accounts.identify_session(cookie) is None
    assert accounts.prune() >= 0


# --- roles --------------------------------------------------------------
def test_the_socket_caller_is_the_owner_without_an_account():
    identity = Identity(kind="socket")
    assert identity.is_owner and identity.user_id is None
    assert identity.name == "owner"


def test_a_plain_account_is_not_an_owner(accounts, guest):
    identity = Identity(kind="token", user=guest)
    assert not identity.is_owner
    assert identity.to_dict()["role"] == "user"


def test_a_role_can_be_changed(accounts, guest):
    assert accounts.set_role("guest", "owner").is_owner
    with pytest.raises(AuthError, match="unknown role"):
        accounts.set_role("guest", "admin")


# --- the rate limiter ---------------------------------------------------
class FakeRequest:
    def __init__(self, host="1.2.3.4"):
        self.client = type("c", (), {"host": host})()


def test_the_rate_limiter_refuses_a_flood():
    from fastapi import HTTPException

    limiter = RateLimiter(per_minute=3)
    for _ in range(3):
        limiter.check(FakeRequest())
    with pytest.raises(HTTPException) as refused:
        limiter.check(FakeRequest())
    assert refused.value.status_code == 429
    # A different caller is unaffected.
    limiter.check(FakeRequest(host="5.6.7.8"))


def test_the_rate_limiter_can_be_switched_off():
    limiter = RateLimiter(per_minute=0)
    for _ in range(50):
        limiter.check(FakeRequest())


# --- chats --------------------------------------------------------------
def test_a_chat_belongs_to_one_account(db, accounts, simon, guest):
    chats = Chats(db)
    mine = chats.create(user_id=simon.id)
    theirs = chats.create(user_id=guest.id)

    assert [c.id for c in chats.all(user_id=simon.id)] == [mine.id]
    with pytest.raises(KeyError):
        chats.get(theirs.id, user_id=simon.id)
    with pytest.raises(KeyError):
        chats.messages(theirs.id, user_id=simon.id)


def test_the_owner_over_the_socket_sees_the_chats_with_no_account(db, accounts, simon):
    chats = Chats(db)
    socket_chat = chats.create(user_id=None)
    chats.create(user_id=simon.id)
    assert [c.id for c in chats.all(user_id=None)] == [socket_chat.id]


def test_a_chat_is_named_after_its_first_message(db):
    chats = Chats(db)
    chat = chats.create(user_id=None)
    chats.title_if_unset(chat.id, "how do I reverse a list in python?")
    assert chats.get(chat.id, user_id=None).title == "how do I reverse a list in python?"
    # And not renamed by the second.
    chats.title_if_unset(chat.id, "something else entirely")
    assert chats.get(chat.id, user_id=None).title.startswith("how do I")


def test_a_long_first_message_is_trimmed():
    assert len(title_from("x" * 200)) <= 60
    assert title_from("   ") == "new chat"


def test_archiving_hides_a_chat_without_losing_the_turns(db):
    chats = Chats(db)
    chat = chats.create(user_id=None)
    db.insert("turns", {"session_id": chat.id, "role": "user", "content": "hello"})
    chats.archive(chat.id, user_id=None)

    assert chats.all(user_id=None) == []
    assert db.query_one("SELECT COUNT(*) AS n FROM turns")["n"] == 1


def test_messages_come_back_with_their_feedback(db):
    chats = Chats(db)
    chat = chats.create(user_id=None)
    turn = db.insert("turns", {"session_id": chat.id, "role": "assistant", "content": "hi"})
    db.insert("feedback", {"turn_id": turn, "kind": "good", "source": "user"})

    messages = chats.messages(chat.id, user_id=None)
    assert messages[0]["feedback"][0]["kind"] == "good"
