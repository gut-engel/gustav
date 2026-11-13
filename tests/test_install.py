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


@pytest.fixture(autouse=True)
def logs_go_somewhere_disposable(tmp_path_factory, monkeypatch):
    """Both scripts keep a log under XDG_STATE_HOME. Not in the home of whoever runs the tests."""
    state = tmp_path_factory.mktemp("state")
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    return state


def logs(state: Path, which: str = "install") -> list[Path]:
    return sorted((state / "gustav").glob(f"{which}-*.log"))


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
    assert "GUSTAV_TOKEN" in result.stdout, "a private fork still needs a way in"


def test_it_refuses_to_guess_when_nothing_can_answer(tmp_path):
    """With no terminal and no mode, asking is impossible — so it stops."""
    result = run("--dir", str(tmp_path / "x"), "--source", str(REPO))
    assert result.returncode != 0
    assert "--server or --client" in result.stderr


def test_the_normal_path_carries_no_credential(tmp_path):
    """The repository is public, so the one-liner is a plain clone and nothing else."""
    result = run(
        "--client", "--dry-run", "--dir", str(tmp_path / "fresh"),
        "--url", "https://example:8713", env={"GUSTAV_TOKEN": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "git clone --progress --branch main --depth 1 https://github.com/gut-engel/gustav.git" \
        in result.stdout
    assert "x-access-token" not in result.stdout


def test_a_token_is_still_honoured_for_a_private_fork_and_never_printed(tmp_path):
    result = run(
        "--client", "--dry-run", "--dir", str(tmp_path / "fresh"),
        "--url", "https://example:8713", env={"GUSTAV_TOKEN": "github_pat_pretend"},
    )
    assert result.returncode == 0, result.stderr
    assert "x-access-token:***@github.com" in result.stdout
    assert "github_pat_pretend" not in result.stdout + result.stderr


def test_the_one_liner_shape_works_when_the_script_itself_is_on_stdin(tmp_path):
    """`curl … | bash -s -- --client` must not read the rest of itself as an answer."""
    result = subprocess.run(
        ["bash", "-s", "--", "--client", "--dry-run", "--source", str(REPO),
         "--dir", str(tmp_path / "piped"), "--url", "https://example:8713"],
        input=INSTALL.read_text(encoding="utf-8"),
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "would run" in result.stdout
    assert not (tmp_path / "piped").exists()


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
def test_the_slow_stage_writes_down_everything_around_it(tmp_path, logs_go_somewhere_disposable):
    """pip is the step that takes ten minutes and goes wrong on a machine you cannot see."""
    result = run("--client", "--source", str(REPO), "--dir", str(tmp_path / "gustav"),
                 env={"GUSTAV_BIN": str(tmp_path / "bin")}, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    text = logs(logs_go_somewhere_disposable)[0].read_text(encoding="utf-8")

    assert "the interpreter and the packaging tools it will use" in text
    assert "room where the wheels are going" in text, "a full disk is the usual reason"
    assert "Using pip" in text, "pip talks into the log at -v"
    assert "files.pythonhosted.org" in text, "which wheel came from where"
    assert "everything in the virtualenv now" in text and "gustav==" in text
    assert "No broken requirements found" in text

    # None of which belongs on the screen.
    assert "files.pythonhosted.org" not in result.stdout
    assert "packages," in result.stdout, "just the count and the size"


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


def test_it_says_where_the_log_is_before_anything_happens(tmp_path, logs_go_somewhere_disposable):
    """The path is on the screen first, so a failure five minutes later has somewhere to point."""
    result = run("--client", "--dry-run", "--source", str(REPO), "--dir", str(tmp_path / "x"),
                 "--url", "https://example:8713")
    assert result.returncode == 0, result.stderr
    written = logs(logs_go_somewhere_disposable)
    assert len(written) == 1, "exactly one log per run"
    assert str(written[0]) in result.stdout.splitlines()[0]


def test_the_log_holds_the_machine_the_commands_and_the_steps(tmp_path, logs_go_somewhere_disposable):
    result = run("--client", "--dry-run", "--source", str(REPO), "--dir", str(tmp_path / "x"),
                 "--url", "https://example:8713")
    assert result.returncode == 0, result.stderr
    text = logs(logs_go_somewhere_disposable)[0].read_text(encoding="utf-8")
    assert "uname" not in text, "the machine is described, not the command that described it"
    assert "machine:" in text and "python" in text
    assert "step 1/6: Checking what is here" in text
    assert "$ " in text, "every command it runs is written down"
    assert "pip install" in text


def test_the_log_never_holds_the_token(tmp_path, logs_go_somewhere_disposable):
    result = run("--client", "--dry-run", "--dir", str(tmp_path / "fresh"),
                 "--url", "https://example:8713", env={"GUSTAV_TOKEN": "github_pat_pretend"})
    assert result.returncode == 0, result.stderr
    text = logs(logs_go_somewhere_disposable)[0].read_text(encoding="utf-8")
    assert "github_pat_pretend" not in text
    assert "x-access-token:***@github.com" in text


def test_a_broken_command_shows_the_end_of_the_log(tmp_path, logs_go_somewhere_disposable):
    """When something actually breaks, the last thing it said is on the screen."""
    broken = tmp_path / "pretend-git"
    broken.mkdir()
    (broken / "git").write_text("#!/bin/sh\necho 'fatal: pretend git is broken' >&2\nexit 2\n",
                                encoding="utf-8")
    (broken / "git").chmod(0o755)
    checkout = tmp_path / "gustav"
    (checkout / ".git").mkdir(parents=True)
    result = run("--client", "--dir", str(checkout),
                 env={"PATH": f"{broken}:{os.environ['PATH']}"})
    assert result.returncode != 0
    assert "could not fetch" in result.stderr
    assert "fatal: pretend git is broken" in result.stderr, "the reason, not just the complaint"


def git_checkout_that_has_drifted(tmp_path) -> Path:
    """An origin, a clone of it, and a commit of your own on top: no fast-forward from here."""
    origin, work = tmp_path / "origin.git", tmp_path / "gustav"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "--quiet", str(origin), str(seed)], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    (seed / "pyproject.toml").write_text("[project]\nname='gustav'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "--quiet", "-m", "first"], check=True, env=env)
    subprocess.run(["git", "-C", str(seed), "push", "--quiet", "origin", "main"], check=True)
    subprocess.run(["git", "clone", "--quiet", str(origin), str(work)], check=True)
    (work / "src.py").write_text("+  not valid python\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(work), "commit", "--quiet", "-m", "mine"], check=True, env=env)
    return work


def test_a_file_deleted_from_the_checkout_is_put_back(tmp_path):
    """The server's actual failure: a fast-forward does not notice a hole in the tree."""
    origin, work = tmp_path / "origin.git", tmp_path / "gustav"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "--quiet", str(origin), str(seed)], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    (seed / "pyproject.toml").write_text("[project]\nname='gustav'\n", encoding="utf-8")
    (seed / "src").mkdir()
    (seed / "src" / "needed.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "--quiet", "-m", "first"], check=True, env=env)
    subprocess.run(["git", "-C", str(seed), "push", "--quiet", "origin", "main"], check=True)
    subprocess.run(["git", "clone", "--quiet", str(origin), str(work)], check=True)

    (work / "src" / "needed.py").unlink()   # the hole a fast-forward will not fill
    result = run("--client", "--dry-run", "--dir", str(work), "--url", "https://example:8713")
    assert result.returncode == 0, result.stderr
    assert "missing from it" in result.stderr and "needed.py" in result.stderr
    assert "would put them back" in result.stdout

    result = run("--client", "--keep-local", "--dry-run", "--dir", str(work),
                 "--url", "https://example:8713")
    assert "leaving them missing" in result.stderr, "--keep-local means what it says"


def test_it_will_not_silently_install_a_checkout_that_has_drifted(tmp_path):
    """The failure that started this: a local edit installed as if it were the release."""
    work = git_checkout_that_has_drifted(tmp_path)
    result = run("--client", "--dir", str(work), "--url", "https://example:8713")
    assert result.returncode != 0
    assert "cannot be brought up to main" in result.stderr
    assert "--reset" in result.stderr and "--keep-local" in result.stderr
    assert (work / "src.py").exists(), "it refused; it did not help itself to anything"


def test_reset_puts_what_is_there_aside(tmp_path):
    work = git_checkout_that_has_drifted(tmp_path)
    result = run("--client", "--dry-run", "--reset", "--dir", str(work),
                 "--url", "https://example:8713")
    assert result.returncode == 0, result.stderr
    assert "stash push" in result.stdout
    assert "reset --hard" in result.stdout


def test_keep_local_installs_what_is_there_and_says_so(tmp_path):
    work = git_checkout_that_has_drifted(tmp_path)
    result = run("--client", "--dry-run", "--keep-local", "--dir", str(work),
                 "--url", "https://example:8713")
    assert result.returncode == 0, result.stderr
    assert "leaving the checkout as it is" in result.stderr
    assert "stash" not in result.stdout


# --- the updater --------------------------------------------------------
UPDATE = REPO / "scripts" / "update.sh"


def update_sh(*args: str, env: dict | None = None, timeout: int = 120):
    return subprocess.run(
        ["bash", str(UPDATE), *args],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, **(env or {})},
    )


def a_checkout_with_an_origin(tmp_path, name: str = "gustav") -> Path:
    """A clone that is exactly its origin — the ordinary case an update starts from."""
    origin, seed, work = tmp_path / "origin.git", tmp_path / "seed", tmp_path / name
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(origin), str(seed)], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    (seed / "pyproject.toml").write_text("[project]\nname='gustav'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "--quiet", "-m", "first"], check=True, env=env)
    subprocess.run(["git", "-C", str(seed), "push", "--quiet", "origin", "main"], check=True)
    subprocess.run(["git", "clone", "--quiet", str(origin), str(work)], check=True)
    return work


def fake_ai(checkout: Path, body: str) -> Path:
    binary = checkout / ".venv" / "bin" / "ai"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(body, encoding="utf-8")
    binary.chmod(0o755)
    return binary


def test_the_updater_is_valid_and_explains_itself():
    assert subprocess.run(["bash", "-n", str(UPDATE)], capture_output=True).returncode == 0
    assert update_sh("--version").stdout.strip().startswith("update.sh 20")
    assert "--check" in update_sh("--help").stdout


def test_updating_something_that_is_not_there_says_so(tmp_path):
    result = update_sh("--dir", str(tmp_path / "nothing"))
    assert result.returncode != 0 and "is not there" in result.stderr


def test_it_will_not_update_what_it_did_not_install(tmp_path):
    """A directory, but no checkout: there is nothing to pull and it says which."""
    (tmp_path / "var").mkdir()
    result = update_sh("--dir", str(tmp_path))
    assert result.returncode != 0 and "not a git checkout" in result.stderr

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    result = update_sh("--dir", str(tmp_path))
    assert result.returncode != 0 and "no virtualenv" in result.stderr


def test_the_updater_hands_over_to_the_command_in_the_package(tmp_path):
    """The shell moves the source; everything after that is 'ai update', which has tests."""
    checkout = a_checkout_with_an_origin(tmp_path)
    said = checkout / "said.txt"
    fake_ai(
        checkout,
        f'#!/bin/sh\nif [ "$2" = --help ]; then exit 0; fi\nprintf \'%s\\n\' "$*" > {said}\n',
    )

    result = update_sh("--dir", str(checkout), "--ref", "main", "--check", "--no-restart")
    assert result.returncode == 0, result.stdout + result.stderr
    handed = said.read_text(encoding="utf-8")
    assert handed.startswith("update --log "), "the command gets the same log, not a second one"
    assert "--ref main" in handed and "--check" in handed and "--no-restart" in handed


def test_an_install_too_old_to_know_the_command_is_told_what_to_run(tmp_path):
    checkout = a_checkout_with_an_origin(tmp_path)
    fake_ai(checkout, "#!/bin/sh\nexit 2\n")

    result = update_sh("--dir", str(checkout), "--ref", "main")
    assert result.returncode != 0
    assert "too old to know" in result.stderr and "install.sh" in result.stderr


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


def test_the_uninstaller_writes_a_log_too(tmp_path, logs_go_somewhere_disposable):
    (tmp_path / "var").mkdir()
    (tmp_path / ".venv").mkdir()
    result = uninstall("--dir", str(tmp_path), "--yes", env={"GUSTAV_BIN": str(tmp_path / "bin")})
    assert result.returncode == 0, result.stderr
    written = logs(logs_go_somewhere_disposable, "uninstall")
    assert len(written) == 1
    text = written[0].read_text(encoding="utf-8")
    assert "step 4/4" in text
    assert str(tmp_path / ".venv") in text, "it writes down what it removed"
    assert str(written[0]) in result.stdout


def test_a_server_install_reads_the_driver_before_it_downloads_anything(tmp_path):
    """torch must match the driver's CUDA, and the driver is the thing that cannot change."""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--query-gpu=name,memory.total" ]; then\n'
        "  echo 'NVIDIA GeForce GTX 1650, 4096 MiB'; exit 0\n"
        "fi\n"
        "echo '| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4   |'\n",
        encoding="utf-8",
    )
    (fake / "nvidia-smi").chmod(0o755)
    result = run("--server", "--dry-run", "--no-service", "--source", str(REPO),
                 "--dir", str(tmp_path / "s"),
                 env={"PATH": f"{fake}:{os.environ['PATH']}"})
    assert result.returncode == 0, result.stderr
    assert "supports CUDA up to 12.4" in result.stdout
    assert "would ask download.pytorch.org" in result.stdout, "and before the 4 GiB, not after"


def test_both_scripts_say_which_version_they_are():
    """raw.githubusercontent.com caches for five minutes; a paste has to say what ran."""
    stamped = run("--version").stdout.strip()
    assert stamped.startswith("install.sh 20")
    assert uninstall("--version").stdout.strip().startswith("uninstall.sh 20")


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


def test_uninstalling_one_install_leaves_another_ones_command_alone(tmp_path):
    """Two checkouts on one machine: removing one must not take away the other's `ai`."""
    other = tmp_path / "other"
    (other / ".venv" / "bin").mkdir(parents=True)
    (other / ".venv" / "bin" / "ai").write_text("#!/bin/sh\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ai").symlink_to(other / ".venv" / "bin" / "ai")

    doomed = tmp_path / "doomed"
    (doomed / ".venv").mkdir(parents=True)
    result = uninstall("--dir", str(doomed), "--yes", env={"GUSTAV_BIN": str(bin_dir)})

    assert result.returncode == 0, result.stderr
    assert (bin_dir / "ai").is_symlink(), "it removed a command belonging to another install"
    assert "leaving it alone" in result.stderr
    assert not (doomed / ".venv").exists()


def test_the_uninstaller_also_works_as_a_one_liner(tmp_path):
    (tmp_path / "var").mkdir()
    (tmp_path / ".venv").mkdir()
    result = subprocess.run(
        ["bash", "-s", "--", "--dir", str(tmp_path), "--yes"],
        input=UNINSTALL.read_text(encoding="utf-8"),
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "GUSTAV_BIN": str(tmp_path / "bin")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / ".venv").exists()
    assert (tmp_path / "var").exists()
