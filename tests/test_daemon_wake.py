"""The wake phase through the running daemon: streaming, commands, redaction, OpenAI API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.daemon import Daemon
from gustav.hw.detect import simulated

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def daemon(project: Path) -> AsyncIterator[Daemon]:
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("small-12gb"))
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown("test finished")


@pytest_asyncio.fixture
async def client(daemon: Daemon) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.AsyncHTTPTransport(uds=str(daemon.socket_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
        yield http


async def sse(client: httpx.AsyncClient, path: str, body: dict) -> list[dict]:
    events: list[dict] = []
    async with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200, await response.aread()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))
    return events


# --- streaming ------------------------------------------------------------
async def test_chat_streams_over_sse(client):
    events = await sse(client, "/chat", {"message": "hello world", "stream": True})
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "chunk")
    assert "hello world" in text


async def test_a_streamed_turn_is_logged_like_any_other(daemon, client):
    events = await sse(client, "/chat", {"message": "remember this", "stream": True})
    done = events[-1]
    assert done["turn_id"]
    rows = daemon.db.query("SELECT role, content FROM turns ORDER BY id")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "remember this"


async def test_non_streaming_still_works(client):
    body = (await client.post("/chat", json={"message": "plain"})).json()
    assert "plain" in body["reply"]


# --- commands -------------------------------------------------------------
async def test_a_command_is_answered_without_the_model(daemon, client):
    body = (await client.post("/chat", json={"message": "/help"})).json()
    assert body["command"] == "help" and "/correct" in body["reply"]
    # No turns were recorded: a command is not a conversation.
    assert daemon.db.scalar("SELECT COUNT(*) FROM turns") == 0


async def test_correct_becomes_top_priority_data(daemon, client):
    await client.post("/chat", json={"message": "what is 2+2?"})
    body = (await client.post("/chat", json={"message": "/correct it is 4"})).json()
    assert body["command"] == "correct"
    row = daemon.db.query_one("SELECT kind, payload FROM feedback")
    assert row["kind"] == "correct" and row["payload"] == "it is 4"


async def test_forget_marks_the_exchange_untrainable(daemon, client):
    await client.post("/chat", json={"message": "something private"})
    await client.post("/chat", json={"message": "/forget"})
    assert daemon.db.scalar("SELECT COUNT(*) FROM turns WHERE trainable = 1") == 0


async def test_commands_work_while_streaming(client):
    events = await sse(client, "/chat", {"message": "/help", "stream": True})
    assert events[0]["type"] == "command"
    assert "/good" in events[0]["reply"]


# --- secrets --------------------------------------------------------------
async def test_a_secret_is_redacted_on_disk_and_never_trained_on(daemon, client):
    body = (
        await client.post(
            "/chat",
            json={"message": "my key is sk-proj-abcdefghijklmnopqrstuvwxyz1234 ok?"},
        )
    ).json()
    assert body["redacted"] == ["openai-key"]

    rows = daemon.db.query("SELECT content, trainable, redacted, exclude_reason FROM turns")
    user = rows[0]
    assert "sk-proj-" not in user["content"]
    assert "[REDACTED:openai-key]" in user["content"]
    assert user["trainable"] == 0 and user["redacted"] == 1
    assert "secret:openai-key" == user["exclude_reason"]
    # Both halves of the exchange are excluded, and no work is queued for them.
    assert all(r["trainable"] == 0 for r in rows)
    assert daemon.journal.pending_work_count() == 0


async def test_the_model_still_sees_the_real_text(daemon, client):
    """Redaction protects the store, not the answer."""
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234"
    body = (await client.post("/chat", json={"message": f"is {secret} valid?"})).json()
    assert secret in body["reply"], "the echo proves the responder got the original"


async def test_redaction_can_be_turned_off(project):
    (project / "config.yaml").write_text(
        "logging:\n  console: false\nschedule:\n  enabled: false\n"
        "inference:\n  responder: echo\n"
        "wake:\n  redact_secrets: false\n",
        encoding="utf-8",
    )
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths, detector=simulated("small-12gb"))
    await instance.start()
    try:
        transport = httpx.AsyncHTTPTransport(uds=str(instance.socket_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
            await http.post(
                "/chat", json={"message": "key sk-proj-abcdefghijklmnopqrstuvwxyz1234"}
            )
        row = instance.db.query_one("SELECT content, trainable FROM turns ORDER BY id")
        assert "sk-proj-" in row["content"], "the operator asked to keep the raw text"
        assert row["trainable"] == 0, "it is still never trained on"
    finally:
        await instance.shutdown("done")


# --- implicit feedback ----------------------------------------------------
async def test_a_rejection_is_queued_for_review_not_training(daemon, client):
    await client.post("/chat", json={"message": "how do I open a file in rust?"})
    await client.post("/chat", json={"message": "no, that's wrong"})
    row = daemon.db.query_one("SELECT kind, reviewed, payload FROM feedback")
    assert row["kind"] == "implicit"
    assert row["reviewed"] == 0, "a guess must wait for a human"
    assert "negative" in row["payload"]


async def test_an_ordinary_follow_up_queues_nothing(daemon, client):
    await client.post("/chat", json={"message": "what is a mutex?"})
    await client.post("/chat", json={"message": "and what about a semaphore?"})
    assert daemon.db.scalar("SELECT COUNT(*) FROM feedback") == 0


# --- the OpenAI-compatible endpoint ---------------------------------------
async def test_openai_completions(client):
    body = (
        await client.post(
            "/v1/chat/completions",
            json={"model": "gustav", "messages": [{"role": "user", "content": "hi there"}]},
        )
    ).json()
    assert body["object"] == "chat.completion"
    assert "hi there" in body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"]
    assert "usage" in body


async def test_openai_streaming(client):
    events = await sse(
        client,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "stream me"}], "stream": True},
    )
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events if "choices" in e
    )
    assert "stream me" in text
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


async def test_openai_without_a_user_message_is_a_400(client):
    response = await client.post(
        "/v1/chat/completions", json={"messages": [{"role": "system", "content": "x"}]}
    )
    assert response.status_code == 400


# --- adapters -------------------------------------------------------------
async def test_adapter_status_is_available(client):
    status = (await client.get("/adapter/status")).json()
    assert "modules" in status


async def test_rolling_back_without_adapters_is_a_clear_error(client):
    response = await client.post("/adapter/rollback", json={"module": "coding"})
    assert response.status_code == 409


# --- models ---------------------------------------------------------------
async def test_the_model_listing_shows_the_ladder(client):
    payload = (await client.get("/models")).json()
    assert payload["ladder"]
    assert any(spec["id"] == "qwen3-4b" for spec in payload["ladder"])


async def test_pulling_an_unknown_model_is_a_404(client):
    response = await client.post("/models/pull", json={"model_id": "llama-900b"})
    assert response.status_code == 404
