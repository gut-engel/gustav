"""Who may ask what, over which listener.

The daemon is reachable two ways and they are not the same: a caller on the Unix socket
is the owner because the filesystem decided that, and a caller over TCP is whoever
their token says and nobody without one. These tests drive both.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.daemon import Daemon
from gustav.server.app import create_app
from gustav.server.auth import COOKIE, Accounts

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def daemon(project: Path, repo_root: Path) -> AsyncIterator[Daemon]:
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    config, paths = load_config(project / "config.yaml")
    # The listener itself is not what these tests are about — binding a port would
    # make them fight each other for it. `identify` decides from the request, so an
    # ASGI transport with a client address is exactly a TCP caller.
    instance = Daemon(config, paths)
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown("test finished")


@pytest_asyncio.fixture
async def over_socket(daemon: Daemon) -> AsyncIterator[httpx.AsyncClient]:
    """A client that reaches the daemon the way the local CLI does."""
    transport = httpx.AsyncHTTPTransport(uds=str(daemon.socket_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
        yield http


@pytest_asyncio.fixture
async def over_tcp(daemon: Daemon) -> AsyncIterator[httpx.AsyncClient]:
    """A client that reaches it the way the network does: no socket, no free pass."""
    transport = httpx.ASGITransport(app=create_app(daemon), client=("10.0.0.9", 51234))
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
        yield http


@pytest.fixture
def accounts(daemon):
    return Accounts(daemon.db)


# --- the socket is the owner --------------------------------------------
async def test_the_socket_needs_no_password(over_socket):
    me = (await over_socket.get("/auth/me")).json()
    assert me["role"] == "owner" and me["via"] == "socket"


async def test_the_socket_reaches_owner_only_endpoints(over_socket):
    assert (await over_socket.get("/status")).status_code == 200
    assert (await over_socket.get("/tools")).status_code == 200


# --- TCP needs an account -----------------------------------------------
async def test_tcp_without_a_token_is_refused(over_tcp):
    response = await over_tcp.get("/status")
    assert response.status_code == 401
    assert "account" in response.json()["detail"]
    assert response.headers["www-authenticate"] == "Bearer"


async def test_health_is_the_one_thing_anyone_may_ask(over_tcp):
    assert (await over_tcp.get("/health")).status_code == 200


async def test_a_token_gets_in_as_its_account(over_tcp, accounts):
    accounts.create("simon", "a-good-password", role="owner")
    token = accounts.issue_token("simon")
    me = (
        await over_tcp.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    ).json()
    assert me["name"] == "simon" and me["role"] == "owner" and me["via"] == "token"


async def test_a_plain_account_may_chat_but_not_review_tools(over_tcp, accounts):
    accounts.create("guest", "another-password")
    token = accounts.issue_token("guest")
    headers = {"Authorization": f"Bearer {token}"}

    assert (await over_tcp.get("/auth/me", headers=headers)).status_code == 200
    chat = await over_tcp.post("/chat", json={"message": "hello"}, headers=headers)
    assert chat.status_code == 200

    for path in ("/status", "/tools", "/backups", "/expansion"):
        refused = await over_tcp.get(path, headers=headers)
        assert refused.status_code == 403, path
        assert "owner" in refused.json()["detail"]


async def test_an_owner_account_reaches_everything(over_tcp, accounts):
    accounts.create("simon", "a-good-password", role="owner")
    token = accounts.issue_token("simon")
    headers = {"Authorization": f"Bearer {token}"}
    assert (await over_tcp.get("/status", headers=headers)).status_code == 200
    assert (await over_tcp.get("/tools", headers=headers)).status_code == 200


async def test_a_revoked_token_stops_working_immediately(over_tcp, accounts, daemon):
    accounts.create("simon", "a-good-password", role="owner")
    token = accounts.issue_token("simon")
    headers = {"Authorization": f"Bearer {token}"}
    assert (await over_tcp.get("/auth/me", headers=headers)).status_code == 200

    accounts.revoke_token(daemon.db.query_one("SELECT id FROM api_tokens")["id"])
    assert (await over_tcp.get("/auth/me", headers=headers)).status_code == 401


# --- signing in ----------------------------------------------------------
async def test_a_password_can_be_exchanged_for_a_token(over_tcp, accounts):
    accounts.create("simon", "a-good-password", role="owner")
    payload = (
        await over_tcp.post(
            "/auth/token",
            json={"name": "simon", "password": "a-good-password", "label": "laptop"},
        )
    ).json()
    assert payload["token"].startswith("gus_")
    assert payload["user"]["name"] == "simon"

    me = (
        await over_tcp.get(
            "/auth/me", headers={"Authorization": f"Bearer {payload['token']}"}
        )
    ).json()
    assert me["name"] == "simon"


async def test_the_wrong_password_gets_nothing(over_tcp, accounts):
    accounts.create("simon", "a-good-password")
    response = await over_tcp.post(
        "/auth/token", json={"name": "simon", "password": "wrong"}
    )
    assert response.status_code == 401


async def test_a_browser_gets_an_httponly_cookie(over_tcp, accounts):
    accounts.create("simon", "a-good-password", role="owner")
    response = await over_tcp.post(
        "/auth/login", json={"name": "simon", "password": "a-good-password"}
    )
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert COOKIE in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie

    # And it works for the next request, without a token.
    assert (await over_tcp.get("/auth/me")).json()["name"] == "simon"
    await over_tcp.post("/auth/logout")
    assert (await over_tcp.get("/auth/me")).status_code == 401


# --- managing accounts ---------------------------------------------------
async def test_accounts_are_made_over_the_socket(over_socket):
    created = (
        await over_socket.post(
            "/users", json={"name": "simon", "password": "a-good-password", "role": "owner"}
        )
    ).json()
    assert created["name"] == "simon" and created["role"] == "owner"

    listing = (await over_socket.get("/users")).json()
    assert [u["name"] for u in listing["users"]] == ["simon"]


async def test_a_short_password_is_refused_by_the_api(over_socket):
    response = await over_socket.post(
        "/users", json={"name": "simon", "password": "short"}
    )
    assert response.status_code == 422


async def test_a_token_can_be_issued_for_another_machine(over_socket):
    await over_socket.post(
        "/users", json={"name": "simon", "password": "a-good-password", "role": "owner"}
    )
    payload = (await over_socket.post("/users/simon", json={"token": "laptop"})).json()
    assert payload["token"].startswith("gus_")


async def test_a_plain_account_cannot_make_accounts(over_tcp, accounts):
    accounts.create("guest", "another-password")
    token = accounts.issue_token("guest")
    response = await over_tcp.post(
        "/users",
        json={"name": "another", "password": "a-good-password"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


async def test_the_service_token_still_works(daemon, accounts):
    """`instance.http_token` predates accounts and is still the simple one-user path."""
    daemon.config.instance.http_token = "a-shared-secret"
    transport = httpx.ASGITransport(app=create_app(daemon), client=("10.0.0.9", 1))
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
        me = (
            await http.get(
                "/auth/me", headers={"Authorization": "Bearer a-shared-secret"}
            )
        ).json()
        assert me["role"] == "owner" and me["via"] == "service"
        assert (await http.get("/auth/me", headers={"Authorization": "Bearer wrong"})).status_code == 401


# --- the listener --------------------------------------------------------
async def test_the_port_is_closed_until_you_open_it(project, repo_root):
    """The default install is not reachable from the network at all."""
    import socket

    from gustav.config.load import load_config as _load

    config, paths = _load(project / "config.yaml")
    assert config.api.enabled is False
    assert config.web.enabled is False

    instance = Daemon(config, paths)
    await instance.start()
    try:
        probe = socket.socket()
        probe.settimeout(1.0)
        refused = probe.connect_ex(("127.0.0.1", config.api.port)) != 0
        probe.close()
        assert refused, f"something is listening on {config.api.port} with api.enabled false"
    finally:
        await instance.shutdown("test finished")


async def test_turning_the_api_on_opens_the_port(project, repo_root):
    import socket

    from gustav.config.load import load_config as _load

    config, paths = _load(project / "config.yaml")
    config.api.enabled = True
    # A port nothing else in the suite uses.
    with socket.socket() as finder:
        finder.bind(("127.0.0.1", 0))
        config.api.port = finder.getsockname()[1]

    instance = Daemon(config, paths)
    await instance.start()
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{config.api.port}"
        ) as http:
            assert (await http.get("/health")).status_code == 200
            # Open does not mean unauthenticated.
            assert (await http.get("/status")).status_code == 401
    finally:
        await instance.shutdown("test finished")
