"""Reviewing tools through the running daemon: show, try, approve, reject."""

from __future__ import annotations

import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.daemon import Daemon
from gustav.tools.review import check
from gustav.tools.spec import ToolSpec
from gustav.tools.store import ToolStore

pytestmark = pytest.mark.asyncio

ADD = ToolSpec(
    name="add_numbers",
    description="Add two numbers together and return the sum.",
    source="def run(a, b):\n    return a + b",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "the first number"},
            "b": {"type": "number", "description": "the second number"},
        },
        "required": ["a", "b"],
    },
    examples=({"arguments": {"a": 2, "b": 3}, "expect": 5},),
)


@pytest_asyncio.fixture
async def daemon(project: Path, repo_root: Path) -> AsyncIterator[Daemon]:
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths)
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


def propose(daemon: Daemon, spec: ToolSpec = ADD):
    store = ToolStore(daemon.db, daemon.paths.tools)
    return store.propose(spec, check(spec, daemon.config.tools))


async def test_a_proposed_tool_is_listed_as_waiting(daemon, client):
    propose(daemon)
    payload = (await client.get("/tools")).json()
    assert payload["enabled"] is True and payload["require_approval"] is True
    assert [t["name"] for t in payload["tools"]] == ["add_numbers"]
    assert payload["tools"][0]["status"] == "proposed"
    assert [p["name"] for p in payload["pending"]] == ["add_numbers"]


async def test_the_review_shows_the_source_the_limits_and_the_checks(daemon, client):
    propose(daemon)
    payload = (await client.get("/tools/add_numbers")).json()

    assert payload["source"] == ADD.source
    assert payload["capabilities"] == ["pure"]
    assert payload["checks"]["ok"] is True
    assert payload["checks"]["examples"][0]["got"] == 5
    assert any("compute only" in line for line in payload["limits"])
    assert payload["version_status"] == "proposed"


async def test_a_tool_can_be_tried_before_it_is_approved(daemon, client):
    propose(daemon)
    payload = (
        await client.post("/tools/add_numbers/run", json={"arguments": {"a": 2, "b": 40}})
    ).json()
    assert payload["ok"] and payload["result"] == 42
    assert payload["status"] == "proposed", "trying it is part of reading it"
    logged = daemon.db.query_one("SELECT source, ok FROM tool_calls")
    assert logged["source"] == "cli" and logged["ok"] == 1


async def test_approving_it_makes_it_callable_and_is_recorded(daemon, client):
    propose(daemon)
    payload = (
        await client.post(
            "/tools/add_numbers/review",
            json={"approve": True, "by": "simon", "note": "read it"},
        )
    ).json()
    assert payload["approved"] and payload["status"] == "approved"

    store = ToolStore(daemon.db, daemon.paths.tools)
    assert [v.name for v in store.approved()] == ["add_numbers"]
    event = daemon.db.query_one("SELECT * FROM events WHERE kind = 'tool.approved'")
    assert "by simon" in event["message"] and "read it" in event["message"]


async def test_rejecting_it_leaves_it_unusable(daemon, client):
    propose(daemon)
    await client.post(
        "/tools/add_numbers/review", json={"approve": False, "note": "no thanks"}
    )
    store = ToolStore(daemon.db, daemon.paths.tools)
    assert store.approved() == []
    assert store.record("add_numbers").status == "rejected"


async def test_a_tool_that_failed_its_checks_cannot_be_approved_over_the_api(daemon, client):
    bad = ToolSpec(
        name="does_nothing",
        description="A tool whose examples prove nothing whatsoever.",
        source="def run(a):\n    return None",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number", "description": "x"}},
            "required": ["a"],
        },
        examples=({"arguments": {"a": 1}, "expect": None},),
    )
    propose(daemon, bad)
    response = await client.post("/tools/does_nothing/review", json={"approve": True})
    assert response.status_code == 400
    assert "admission checks" in response.json()["detail"]


async def test_requesting_a_tool_queues_it_for_the_night(daemon, client):
    payload = (
        await client.post("/tools/requests", json={"text": "convert between units"})
    ).json()
    assert payload["status"] == "open"
    listing = (await client.get("/tools")).json()
    assert [r["text"] for r in listing["requests"]] == ["convert between units"]
    # It counts as pending work, which is one of the things that triggers a sleep.
    assert daemon.journal.pending_work_count() >= 1


async def test_an_empty_request_is_refused(daemon, client):
    response = await client.post("/tools/requests", json={"text": "   "})
    assert response.status_code == 400


async def test_retiring_and_rolling_back(daemon, client):
    propose(daemon)
    await client.post("/tools/add_numbers/review", json={"approve": True})
    better = ToolSpec(
        **{**ADD.__dict__, "source": "def run(a, b):\n    return float(a) + float(b)",
           "examples": ({"arguments": {"a": 2, "b": 3}, "expect": 5.0},)}
    )
    propose(daemon, better)
    await client.post("/tools/add_numbers/review", json={"approve": True, "version": "v002"})

    rolled = (await client.post("/tools/add_numbers/rollback")).json()
    assert rolled["version"] == "v001"

    retired = (await client.post("/tools/add_numbers/retire", json={"note": "not now"})).json()
    assert retired["status"] == "retired"
    assert ToolStore(daemon.db, daemon.paths.tools).approved() == []


async def test_deleting_a_tool_removes_its_source(daemon, client):
    propose(daemon)
    payload = (await client.request("DELETE", "/tools/add_numbers")).json()
    assert payload["deleted"]
    assert not (daemon.paths.tools / "add_numbers").exists()
    assert (await client.get("/tools")).json()["tools"] == []


async def test_an_unknown_tool_is_a_404(daemon, client):
    assert (await client.get("/tools/nope")).status_code == 404


async def test_tools_are_in_the_backup_set(daemon, client):
    propose(daemon)
    result = daemon.create_backup("manual")
    assert "tools" in json.dumps(result)


# --- asking for one from the chat box ------------------------------------
# "make me a tool for that" is said in the middle of a conversation, to a model that
# cannot file anything itself. /tool is the path that does not depend on the model
# understanding the request — it never reaches the model at all.
async def test_asking_for_a_tool_in_the_chat_box_queues_it(daemon, client):
    answer = (
        await client.post(
            "/chat", json={"message": "/tool raise a number to a large power exactly",
                           "stream": False}
        )
    ).json()
    assert answer["command"] == "tool_requested"
    assert "raise a number to a large power exactly" in answer["reply"]

    listing = (await client.get("/tools")).json()
    assert [r["text"] for r in listing["requests"]] == [
        "raise a number to a large power exactly"
    ]
    assert daemon.journal.pending_work_count() >= 1


async def test_the_bare_command_says_what_to_type(daemon, client):
    answer = (await client.post("/chat", json={"message": "/tool", "stream": False})).json()
    assert answer["command"] == "tool_help"
    assert "/tool convert a unix timestamp" in answer["reply"]


async def test_the_chat_box_can_ask_what_it_can_call(daemon, client):
    propose(daemon)
    daemon.review_tool("add_numbers", approve=True, by="simon")
    answer = (await client.post("/chat", json={"message": "/tools", "stream": False})).json()
    assert "add_numbers — Add two numbers together" in answer["reply"]
    assert "/tool <what it should do>" in answer["reply"]


async def test_what_it_says_it_can_call_is_what_it_may_call(daemon):
    """The prompt's list and the approval table are the same fact, twice."""
    propose(daemon)
    assert daemon.capabilities().callable_tools == ()
    assert daemon.capabilities().pending_tools[0].name == "add_numbers"

    daemon.review_tool("add_numbers", approve=True, by="simon")
    after = daemon.capabilities()
    assert [tool.name for tool in after.callable_tools] == ["add_numbers"]
    assert after.pending_tools == ()


async def test_a_requested_tool_is_in_what_the_model_is_told(daemon):
    daemon.request_tool("convert between units")
    from gustav.wake.capabilities import system_prompt

    assert "convert between units" in system_prompt(daemon.capabilities())
