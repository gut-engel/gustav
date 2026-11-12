"""The installer.

The one-liner is the first thing anyone runs, so the script is tested the way it will
be used: a dry run that must touch nothing, and a real client install into a throwaway
directory that must end with a working `ai`.

The server path is not installed here — it pulls four gigabytes of CUDA wheels — but
everything up to that point is shared, and the dry run covers which command it would
have run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALL = REPO / "scripts" / "install.sh"


def run(*args: str, env: dict | None = None, timeout: int = 600):
    return subprocess.run(
        ["bash", str(INSTALL), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )


def test_the_script_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", str(INSTALL)], capture_output=True).returncode == 0


def test_help_explains_both_modes():
    result = run("--help")
    assert result.returncode == 0
    assert "--server" in result.stdout and "--client" in result.stdout
    assert "GUSTAV_TOKEN" in result.stdout


def test_it_refuses_to_guess_when_nothing_can_answer(tmp_path):
    """With no terminal and no mode, asking is impossible — so it stops."""
    result = run("--dir", str(tmp_path / "x"), "--source", str(REPO))
    assert result.returncode != 0
    assert "--server or --client" in result.stderr


def test_a_private_clone_without_a_credential_is_refused(tmp_path):
    result = run("--client", "--dir", str(tmp_path / "fresh"), env={"GUSTAV_TOKEN": ""})
    assert result.returncode != 0
    assert "private" in result.stderr and "GUSTAV_TOKEN" in result.stderr


def test_a_dry_run_changes_nothing(tmp_path):
    target = tmp_path / "nothing-here"
    result = run(
        "--client", "--dry-run", "--source", str(REPO), "--dir", str(target), "--url",
        "https://example:8713",
    )
    assert result.returncode == 0, result.stderr
    assert not target.exists()
    assert "would run" in result.stdout


def test_the_dry_run_says_which_wheels_it_would_install(tmp_path):
    server = run("--server", "--dry-run", "--no-service", "--source", str(REPO),
                 "--dir", str(tmp_path / "s"))
    client = run("--client", "--dry-run", "--source", str(REPO),
                 "--dir", str(tmp_path / "c"))
    assert "[ml]" in server.stdout, "the server install pulls the ML extras"
    assert "[ml]" not in client.stdout, "a client has no business downloading CUDA"
    assert "no torch" in client.stdout


def test_it_refuses_to_install_over_something_else(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "important.txt").write_text("do not delete me", encoding="utf-8")
    result = run("--client", "--dir", str(occupied))
    assert result.returncode != 0
    assert "not a git checkout" in result.stderr
    assert (occupied / "important.txt").exists()


@pytest.mark.slow
def test_a_client_install_produces_a_working_ai(tmp_path):
    """The whole point: one command, and `ai` works."""
    target = tmp_path / "gustav"
    bin_dir = tmp_path / "bin"
    result = run(
        "--client",
        "--source", str(REPO),
        "--dir", str(target),
        env={"GUSTAV_BIN": str(bin_dir)},
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assert (target / ".venv" / "bin" / "ai").exists()
    assert (bin_dir / "ai").is_symlink()

    # It is a client install: no torch, however long that would have taken.
    assert not (target / ".venv" / "lib").joinpath(
        f"python3.{sys.version_info.minor}", "site-packages", "torch"
    ).exists()

    # And the command actually runs.
    version = subprocess.run(
        [str(bin_dir / "ai"), "--help"], capture_output=True, text=True, timeout=120
    )
    assert version.returncode == 0
    assert "login" in version.stdout

    # With no daemon and no saved server, it says so rather than hanging.
    status = subprocess.run(
        [str(bin_dir / "ai"), "status"],
        capture_output=True, text=True, timeout=120, cwd=str(tmp_path),
    )
    assert status.returncode == 3
    assert "no daemon" in status.stderr


@pytest.mark.slow
def test_installing_again_updates_in_place(tmp_path):
    target = tmp_path / "gustav"
    bin_dir = tmp_path / "bin"
    first = run("--client", "--source", str(REPO), "--dir", str(target),
                env={"GUSTAV_BIN": str(bin_dir)}, timeout=900)
    assert first.returncode == 0, first.stderr
    marker = target / "var" / "keep-me.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("still here", encoding="utf-8")

    second = run("--client", "--source", str(REPO), "--dir", str(target),
                 env={"GUSTAV_BIN": str(bin_dir)}, timeout=900)
    assert second.returncode == 0, second.stderr
    assert marker.read_text(encoding="utf-8") == "still here"


# --- the uninstaller ----------------------------------------------------
UNINSTALL = REPO / "scripts" / "uninstall.sh"


def uninstall(*args: str, env: dict | None = None, timeout: int = 120):
    return subprocess.run(
        ["bash", str(UNINSTALL), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )


def test_the_uninstaller_is_valid_and_explains_itself():
    assert subprocess.run(["bash", "-n", str(UNINSTALL)], capture_output=True).returncode == 0
    assert "--purge" in uninstall("--help").stdout


def test_uninstalling_something_that_is_not_there_says_so(tmp_path):
    result = uninstall("--dir", str(tmp_path / "nothing"), "--yes")
    assert result.returncode != 0 and "is not there" in result.stderr


def test_it_refuses_to_proceed_unasked_with_nothing_to_confirm_with(tmp_path):
    (tmp_path / "var").mkdir()
    result = uninstall("--dir", str(tmp_path))
    assert result.returncode != 0 and "--yes" in result.stderr


def test_what_it_learned_survives_an_ordinary_uninstall(tmp_path):
    for part in ("var", "adapters", "tools", "backups"):
        (tmp_path / part).mkdir()
        (tmp_path / part / "keep").write_text("mine", encoding="utf-8")
    (tmp_path / ".venv").mkdir()

    result = uninstall("--dir", str(tmp_path), "--yes", env={"GUSTAV_BIN": str(tmp_path / "bin")})
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / ".venv").exists(), "the virtualenv is the installer's to remove"
    for part in ("var", "adapters", "tools", "backups"):
        assert (tmp_path / part / "keep").exists(), f"{part} was not the installer's to delete"
    assert "still in" in result.stdout


def test_purge_says_what_it_will_delete_and_then_does(tmp_path):
    for part in ("var", "adapters", "tools"):
        (tmp_path / part).mkdir()
        (tmp_path / part / "data").write_text("x", encoding="utf-8")

    preview = uninstall("--dir", str(tmp_path), "--purge", "--dry-run", "--yes")
    assert "the database, what it learned" in preview.stderr + preview.stdout
    assert (tmp_path / "var" / "data").exists(), "a dry run deletes nothing"

    result = uninstall("--dir", str(tmp_path), "--purge", "--yes",
                       env={"GUSTAV_BIN": str(tmp_path / "bin")})
    assert result.returncode == 0, result.stderr
    for part in ("var", "adapters", "tools"):
        assert not (tmp_path / part).exists()
