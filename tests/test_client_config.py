"""Where the CLI remembers which server it talks to."""

from __future__ import annotations

import json
import os
import stat

import pytest

from gustav import client


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv(client.ENV_URL, raising=False)
    monkeypatch.delenv(client.ENV_TOKEN, raising=False)
    return tmp_path


def test_with_nothing_saved_it_uses_the_local_socket():
    endpoint = client.resolve()
    assert not endpoint.remote
    assert "socket" in endpoint.describe()


def test_a_saved_server_is_remembered(config_home):
    path = client.save("https://box.example:8713/", "gus_secret", account="simon")
    assert path == config_home / "gustav" / "client.json"

    endpoint = client.resolve()
    assert endpoint.url == "https://box.example:8713", "the trailing slash is dropped"
    assert endpoint.token == "gus_secret" and endpoint.account == "simon"
    assert endpoint.remote


def test_the_file_is_not_readable_by_anyone_else(config_home):
    """It holds a token, so it is not a world-readable file."""
    path = client.save("https://box:8713", "gus_secret")
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_the_token_is_never_in_what_is_printed(config_home):
    client.save("https://box:8713", "gus_very_secret", account="simon")
    assert "gus_very_secret" not in client.resolve().describe()


def test_a_flag_beats_the_environment_beats_the_file(config_home, monkeypatch):
    client.save("https://from-file:8713", "file-token")
    assert client.resolve().url == "https://from-file:8713"

    monkeypatch.setenv(client.ENV_URL, "https://from-env:8713")
    assert client.resolve().url == "https://from-env:8713"
    assert client.resolve().source == client.ENV_URL

    assert client.resolve(url="https://from-flag:8713").url == "https://from-flag:8713"


def test_a_token_flag_applies_to_a_saved_server(config_home):
    client.save("https://box:8713", "old-token")
    assert client.resolve(token="new-token").token == "new-token"


def test_forgetting_is_idempotent(config_home):
    client.save("https://box:8713", "gus_secret")
    assert client.forget() is True
    assert client.forget() is False
    assert not client.resolve().remote


def test_a_corrupt_file_is_ignored_rather_than_fatal(config_home):
    path = config_home / "gustav" / "client.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json at all", encoding="utf-8")
    assert not client.resolve().remote

    path.write_text(json.dumps(["a", "list"]), encoding="utf-8")
    assert not client.resolve().remote


# --- the CLI surface ----------------------------------------------------
def test_bare_ai_is_a_chat():
    from gustav.cli import build_parser

    args = build_parser().parse_args([])
    assert args.command is None


def test_login_and_logout_need_no_daemon():
    from gustav.cli import build_parser

    for command in ("login", "logout"):
        args = build_parser().parse_args([command] if command == "logout" else [command, "x"])
        assert args.local is True, f"{command} must not need a running daemon"


def test_logout_says_so_when_there_is_nothing_to_forget(capsys):
    from gustav.cli import cmd_logout

    assert cmd_logout(None) == 0
    assert "not signed in" in capsys.readouterr().out


def test_login_without_a_url_explains_itself(capsys):
    import argparse

    from gustav.cli import cmd_login

    args = argparse.Namespace(url=None, token=None, user=None, label="cli")
    assert cmd_login(args) == 2
    assert "ai login <url>" in capsys.readouterr().err
