"""The web interface: what it serves, and who it serves it to.

The page itself is three static files, so there is not much to test about them beyond
"they are in the package and the daemon hands them over when you asked it to". What is
worth testing is everything underneath: the chat endpoints the page drives, the
ownership rule that keeps one account's conversations out of another's browser, and the
fact that switching the interface on does not quietly shadow the API it talks to.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.daemon import Daemon
from gustav.server.app import create_app
from gustav.server.auth import Accounts

pytestmark = pytest.mark.asyncio

WEB = Path(__file__).resolve().parent.parent / "src" / "gustav" / "web"


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


def client(daemon: Daemon, *, web: bool = True, tcp: bool = True) -> httpx.AsyncClient:
    """An app built the way a given config would build it.

    The mount happens inside ``create_app``, so the switch has to be flipped before the
    app exists — which is exactly what restarting the daemon does in real life.
    """
    daemon.config.web.enabled = web
    transport = httpx.ASGITransport(
        app=create_app(daemon), client=("10.0.0.9", 51234) if tcp else None
    )
    return httpx.AsyncClient(transport=transport, base_url="http://gustav")


@pytest_asyncio.fixture
async def owner(daemon: Daemon) -> AsyncIterator[httpx.AsyncClient]:
    """The owner, over the socket: no client address, no password."""
    async with client(daemon, tcp=False) as http:
        yield http


# --- the files -----------------------------------------------------------
async def test_the_three_files_ship_with_the_package():
    assert (WEB / "index.html").is_file()
    assert (WEB / "app.js").is_file()
    assert (WEB / "styles.css").is_file()


async def test_the_page_only_asks_for_files_that_exist():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "styles.css"):
        assert name in html, f"{name} is not referenced by the page"
    # Everything is served from the daemon: a self-improving assistant that is meant to
    # keep working offline must not fetch a framework from someone else's CDN.
    outside = re.sub(r"<svg[^>]*>", "", html)
    assert "http://" not in outside and "https://" not in outside


async def test_the_page_never_builds_html_from_text():
    """The only defence against a conversation scripting the page is not to use innerHTML."""
    source = (WEB / "app.js").read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    assert "innerHTML" not in code
    assert "insertAdjacentHTML" not in code
    assert "document.write" not in code


# --- serving -------------------------------------------------------------
async def test_the_interface_is_off_by_default(daemon):
    async with client(daemon, web=False) as http:
        assert (await http.get("/")).status_code == 404
        assert (await http.get("/app.js")).status_code == 404


async def test_switching_it_on_serves_the_page(daemon):
    async with client(daemon) as http:
        page = await http.get("/")
        assert page.status_code == 200
        assert "<title>gustav</title>" in page.text
        assert (await http.get("/app.js")).status_code == 200
        assert (await http.get("/styles.css")).status_code == 200


async def test_the_page_does_not_shadow_the_api(daemon):
    """The mount is last on purpose: a static file must never win over an endpoint."""
    async with client(daemon) as http:
        assert (await http.get("/health")).status_code == 200
        # And an owner-only endpoint is still owner-only with the interface on.
        assert (await http.get("/status")).status_code == 401


async def test_the_page_can_ask_what_it_is_before_signing_in(daemon):
    async with client(daemon) as http:
        payload = (await http.get("/web/config")).json()
    assert payload["title"] == "gustav"
    assert payload["enabled"] is True
    assert payload["accounts"] is False or payload["accounts"] is True
    assert "owner_panels" in payload


# --- chats ---------------------------------------------------------------
async def test_a_chat_round_trip(owner):
    chat = (await owner.post("/chats", json={"title": ""})).json()
    assert chat["title"] == "new chat"

    sent = await owner.post(
        f"/chats/{chat['id']}/messages", json={"message": "hello there", "stream": False}
    )
    assert sent.status_code == 200

    payload = (await owner.get(f"/chats/{chat['id']}/messages")).json()
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["user", "assistant"]
    assert payload["messages"][0]["content"] == "hello there"
    # The chat names itself after the first thing you said.
    assert payload["chat"]["title"] == "hello there"


async def test_the_chat_list_is_newest_first(owner):
    first = (await owner.post("/chats", json={})).json()
    second = (await owner.post("/chats", json={})).json()
    await owner.post(f"/chats/{first['id']}/messages", json={"message": "later", "stream": False})
    listing = (await owner.get("/chats")).json()
    assert [c["id"] for c in listing["chats"]][0] == first["id"]
    assert second["id"] in [c["id"] for c in listing["chats"]]


async def test_a_chat_streams_the_way_the_page_reads_it(owner):
    chat = (await owner.post("/chats", json={})).json()
    events = []
    async with owner.stream(
        "POST", f"/chats/{chat['id']}/messages", json={"message": "stream me", "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    kinds = [event["type"] for event in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "chunk")


async def test_a_failure_mid_stream_reaches_the_page(owner, daemon, monkeypatch):
    """A dead socket makes the browser invent a reason. The daemon has the real one."""
    chat = (await owner.post("/chats", json={})).json()

    async def refuse(message, *, session=None):
        yield {"type": "start", "session": session, "sleep_paused": False, "redacted": []}
        raise RuntimeError("the weights for qwen3-0.6b are not on this machine")

    monkeypatch.setattr(daemon, "chat_stream", refuse)
    async with owner.stream(
        "POST", f"/chats/{chat['id']}/messages", json={"message": "hi", "stream": True}
    ) as response:
        assert response.status_code == 200
        payloads = [
            json.loads(line[6:])
            async for line in response.aiter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]

    assert payloads[-1]["type"] == "error"
    assert "not on this machine" in payloads[-1]["error"], "the reason, not just a type"


async def test_a_command_in_the_box_is_answered_without_the_model(owner):
    chat = (await owner.post("/chats", json={})).json()
    async with owner.stream(
        "POST", f"/chats/{chat['id']}/messages", json={"message": "/help", "stream": True}
    ) as response:
        payloads = [
            json.loads(line[6:])
            async for line in response.aiter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    assert payloads[0]["type"] == "command"
    assert "/good" in payloads[0]["reply"]


async def test_the_thumbs_reach_the_same_place_the_terminal_does(owner, daemon):
    chat = (await owner.post("/chats", json={})).json()
    await owner.post(f"/chats/{chat['id']}/messages", json={"message": "hi", "stream": False})

    good = await owner.post(f"/chats/{chat['id']}/feedback", json={"kind": "good"})
    assert good.status_code == 200
    row = daemon.db.query_one("SELECT kind FROM feedback ORDER BY id DESC")
    assert row["kind"] == "good"

    fixed = await owner.post(
        f"/chats/{chat['id']}/feedback", json={"kind": "correct", "text": "say it like this"}
    )
    assert fixed.status_code == 200
    row = daemon.db.query_one("SELECT kind, payload FROM feedback ORDER BY id DESC")
    assert row["kind"] == "correct" and "say it like this" in row["payload"]

    # And the page can see them again when it reloads the thread.
    messages = (await owner.get(f"/chats/{chat['id']}/messages")).json()["messages"]
    kinds = [item["kind"] for message in messages for item in message["feedback"]]
    assert "good" in kinds and "correct" in kinds


async def test_a_nonsense_command_is_a_400(owner):
    chat = (await owner.post("/chats", json={})).json()
    response = await owner.post(f"/chats/{chat['id']}/feedback", json={"kind": "nonsense"})
    assert response.status_code in (400, 422)


async def test_archiving_hides_a_chat_but_keeps_what_it_taught(owner, daemon):
    chat = (await owner.post("/chats", json={})).json()
    await owner.post(f"/chats/{chat['id']}/messages", json={"message": "keep me", "stream": False})
    assert (await owner.delete(f"/chats/{chat['id']}")).status_code == 200

    assert [c["id"] for c in (await owner.get("/chats")).json()["chats"]] == []
    turns = daemon.db.query("SELECT id FROM turns WHERE session_id = ?", (chat["id"],))
    assert len(turns) == 2, "archiving must not delete what the night learns from"


# --- one account cannot read another's ------------------------------------
async def test_chats_are_private_to_their_account(daemon):
    accounts = Accounts(daemon.db)
    accounts.create("ada", "a-good-password")
    accounts.create("bob", "another-password")
    ada = {"Authorization": f"Bearer {accounts.issue_token('ada')}"}
    bob = {"Authorization": f"Bearer {accounts.issue_token('bob')}"}

    async with client(daemon) as http:
        chat = (await http.post("/chats", json={}, headers=ada)).json()
        await http.post(
            f"/chats/{chat['id']}/messages",
            json={"message": "something private", "stream": False},
            headers=ada,
        )

        assert (await http.get("/chats", headers=bob)).json()["chats"] == []
        assert (await http.get(f"/chats/{chat['id']}/messages", headers=bob)).status_code == 404
        assert (await http.delete(f"/chats/{chat['id']}", headers=bob)).status_code == 404
        assert (
            await http.post(
                f"/chats/{chat['id']}/messages",
                json={"message": "hello", "stream": False},
                headers=bob,
            )
        ).status_code == 404
        # Ada still has hers.
        assert len((await http.get("/chats", headers=ada)).json()["chats"]) == 1


async def test_the_owners_own_chats_are_not_an_accounts(daemon):
    """A chat made over the socket belongs to the machine's owner, not to an account."""
    accounts = Accounts(daemon.db)
    accounts.create("ada", "a-good-password")
    headers = {"Authorization": f"Bearer {accounts.issue_token('ada')}"}

    async with client(daemon, tcp=False) as socket_client:
        mine = (await socket_client.post("/chats", json={})).json()
    async with client(daemon) as http:
        assert (await http.get(f"/chats/{mine['id']}/messages", headers=headers)).status_code == 404


async def test_a_signed_out_browser_gets_nothing(daemon):
    async with client(daemon) as http:
        assert (await http.get("/chats")).status_code == 401
        assert (await http.post("/chats", json={})).status_code == 401


# --- the panels' contract -------------------------------------------------
async def test_the_owner_panels_get_the_keys_the_page_reads(owner, daemon):
    """A silent rename on the daemon side would leave the panels blank, not broken.

    The page is not type-checked against the API, so this is the check that it is
    reading fields that exist. Each name here appears in `app.js`.
    """
    status = (await owner.get("/status")).json()
    assert {"state", "uptime_seconds", "learning_paused", "version", "model"} <= status.keys()
    assert {"base_model", "responder"} & status["model"].keys()

    facts = (await owner.get("/facts")).json()
    assert {"facts", "total", "retracted"} <= facts.keys()

    skills = (await owner.get("/skills")).json()
    assert {"skills", "summary"} <= skills.keys()
    assert {"topics", "mean_score"} <= skills["summary"].keys()

    tools = (await owner.get("/tools")).json()
    assert {"tools", "requests"} <= tools.keys()

    expansion = (await owner.get("/expansion")).json()
    assert {"history", "triggers"} <= expansion.keys()

    optimisation = (await owner.get("/optimisation")).json()
    assert {"history", "next_trials"} <= optimisation.keys()

    approvals = (await owner.get("/approvals")).json()
    assert "approvals" in approvals


async def test_a_fact_arrives_in_the_shape_the_table_expects(owner, daemon):
    from gustav.learn.facts import Fact, FactStore

    FactStore(daemon.db).add(
        Fact("the user's editor is helix", subject="editor", confidence=0.9, source="chat")
    )
    row = (await owner.get("/facts")).json()["facts"][0]
    assert {"subject", "text", "confidence", "module", "source", "updated_at"} <= row.keys()


async def test_the_controls_on_the_system_panel_exist(owner):
    """The four buttons under 'State' post to these, and nothing else."""
    for path in ("/control/sleep", "/control/wake", "/control/release-gpu"):
        assert (await owner.post(path, json={})).status_code == 200
    toggled = await owner.post("/control/learning", json={"paused": True})
    assert toggled.status_code == 200
    await owner.post("/control/learning", json={"paused": False})
