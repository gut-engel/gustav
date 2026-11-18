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
@pytest.mark.timeout(900)
def test_a_client_install_produces_a_working_ai(tmp_path):
    """The whole point: one command, and `ai` works.

    It builds a virtualenv and installs the package from PyPI for real, which is four
    minutes on an ordinary connection — well past the 120s this suite's default
    timeout allows. The subprocess already had its own 900s limit; this makes pytest
    agree with it instead of killing a test that was working.
    """
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
@pytest.mark.timeout(900)
def test_the_slow_stage_writes_down_everything_around_it(tmp_path, logs_go_somewhere_disposable):
    """pip is the step that takes ten minutes and goes wrong on a machine you cannot see.

    Which is also why it needs its own timeout: it installs for real, so on a cold pip
    cache it is minutes, and the suite's 120s default was killing a working test.
    """
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
@pytest.mark.timeout(900)   # two real installs, back to back
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


# ---------------------------------------------------------------------------
# the driver installer
#
# It is the only thing in the project that asks for root, so what is tested is mostly
# what it refuses to do. Every distribution is faked through two environment variables
# the script reads for exactly this reason: one machine cannot be six distributions,
# and the per-distribution choice *is* the content of the script.
# ---------------------------------------------------------------------------
DRIVER = REPO / "scripts" / "driver.sh"

OS_RELEASES = {
    "ubuntu": 'ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04"\n',
    "debian": 'ID=debian\nVERSION_ID="12"\nPRETTY_NAME="Debian 12"\n',
    "fedora": 'ID=fedora\nVERSION_ID=41\nPRETTY_NAME="Fedora Linux 41"\n',
    "rocky": 'ID="rocky"\nID_LIKE="rhel centos"\nVERSION_ID="9.4"\nPRETTY_NAME="Rocky 9.4"\n',
    "arch": 'ID=arch\nPRETTY_NAME="Arch Linux"\n',
    "suse": 'ID="opensuse-tumbleweed"\nID_LIKE="opensuse suse"\nVERSION_ID="20261115"\n',
    "void": 'ID=void\nPRETTY_NAME="void"\n',
}


def driver_run(*args: str, distro: str = "ubuntu", env: dict | None = None,
               tmp_path: Path | None = None, timeout: int = 120):
    """The driver installer, told which machine it is standing on."""
    release = tmp_path / f"os-release-{distro}"
    release.write_text(OS_RELEASES[distro], encoding="utf-8")
    return subprocess.run(
        ["bash", str(DRIVER), *args],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "GUSTAV_OS_RELEASE": str(release), **(env or {})},
    )


def test_the_driver_installer_is_valid_and_explains_itself():
    assert subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True).returncode == 0
    help_text = subprocess.run(
        ["bash", str(DRIVER), "--help"], capture_output=True, text=True
    ).stdout
    assert "--toolkit" in help_text
    assert "--dry-run" in help_text
    assert subprocess.run(
        ["bash", str(DRIVER), "--version"], capture_output=True, text=True
    ).stdout.startswith("driver.sh 20")


@pytest.mark.parametrize(
    ("distro", "expected"),
    [
        ("ubuntu", "cuda-drivers"),
        ("debian", "cuda-drivers"),
        ("fedora", "akmod-nvidia"),
        ("rocky", "cuda-drivers"),
        ("arch", "pacman -S"),
        ("suse", "nvidia-video-G06"),
        ("void", "NVIDIA-Linux-x86_64"),
    ],
)
def test_each_distribution_gets_the_driver_its_maintainers_ship(distro, expected, tmp_path):
    result = driver_run("--dry-run", "--yes", distro=distro, tmp_path=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


def test_a_dry_run_changes_nothing_and_says_everything(tmp_path):
    result = driver_run("--dry-run", "--yes", tmp_path=tmp_path)
    assert result.returncode == 0
    assert "would run: sudo apt-get" in result.stdout
    # Both routes, because a real run only learns which applies when it gets there.
    assert "ubuntu2404" in result.stdout
    assert "ubuntu-drivers install" in result.stdout
    assert "nothing was changed" in result.stdout.lower() or "nothing to check" in result.stdout


def test_the_toolkit_is_opt_in(tmp_path):
    without = driver_run("--dry-run", "--yes", tmp_path=tmp_path).stdout
    with_it = driver_run("--dry-run", "--yes", "--toolkit", tmp_path=tmp_path).stdout
    assert "cuda-toolkit" not in without
    assert "not the CUDA toolkit" in without
    assert "cuda-toolkit" in with_it


def test_it_refuses_on_macos_and_says_what_does_work(tmp_path):
    result = driver_run("--dry-run", "--yes", env={"GUSTAV_FAKE_SYSTEM": "Darwin"},
                        tmp_path=tmp_path)
    assert result.returncode != 0
    assert "no NVIDIA driver for macOS" in result.stderr
    assert "--client" in result.stderr, "the thing that does work on that machine"


def test_it_refuses_under_wsl_where_the_driver_belongs_to_windows(tmp_path):
    result = driver_run(
        "--dry-run", "--yes",
        env={"GUSTAV_FAKE_KERNEL": "5.15.153.1-microsoft-standard-WSL2"},
        tmp_path=tmp_path,
    )
    assert result.returncode != 0
    assert "this is WSL" in result.stderr
    assert "winget" in result.stderr
    assert "wsl --shutdown" in result.stderr


def test_it_will_not_install_a_driver_with_nothing_to_confirm_with(tmp_path):
    """Not a dry run, no --yes, no terminal: it must stop before touching anything."""
    result = driver_run("--log", str(tmp_path / "d.log"), tmp_path=tmp_path)
    assert result.returncode != 0
    assert "--yes" in result.stderr or "--dry-run" in result.stderr
    assert "apt-get install" not in (tmp_path / "d.log").read_text(encoding="utf-8")


def test_it_writes_down_what_it_did(tmp_path):
    log = tmp_path / "driver.log"
    driver_run("--dry-run", "--yes", "--log", str(log), tmp_path=tmp_path)
    text = log.read_text(encoding="utf-8")
    assert "driver.sh 20" in text
    assert "step 1/5" in text
    assert "nvidia repository path would be ubuntu2404" in text


def test_the_script_and_the_module_agree_on_every_family():
    """Two copies of the same knowledge; this is the thread that ties them together."""
    from gustav.hw import driver as module

    script = DRIVER.read_text(encoding="utf-8")
    for release in ("ubuntu", "fedora", "rocky", "arch", "suse"):
        platform = module.detect_platform(
            system="Linux", os_release=OS_RELEASES[release], kernel="6.8.0-generic",
            container="", secure_boot=False,
        )
        recipe = module.driver_recipe(platform)
        assert f"install_{recipe.family}()" in script, recipe.family
        assert recipe.tool and recipe.tool in script, (
            f"{recipe.family}: ai doctor says {recipe.tool}, driver.sh does not use it"
        )


# ---------------------------------------------------------------------------
# the satellite installer
#
# One file with no dependencies, so the installer is four small steps: check the sound
# card, fetch the file, write a config, install a service. What is tested is that it
# refuses rather than guessing, and that it never writes a token into a world-readable
# file.
# ---------------------------------------------------------------------------
SATELLITE = REPO / "scripts" / "satellite.sh"


def satellite(*args: str, env: dict | None = None, timeout: int = 120):
    # --skip-sound-check because the machine running the suite is not the machine the
    # microphone is plugged into; everything else about the install is the same.
    return subprocess.run(
        ["bash", str(SATELLITE), "--skip-sound-check", *args],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, **(env or {})},
        stdin=subprocess.DEVNULL,
    )


def test_the_satellite_installer_is_valid_and_explains_itself():
    assert subprocess.run(["bash", "-n", str(SATELLITE)], capture_output=True).returncode == 0
    help_text = satellite("--help").stdout
    assert "--calibrate" in help_text
    assert "--token" in help_text
    assert satellite("--version").stdout.strip().startswith("satellite.sh 20")


def test_it_will_not_guess_which_server_to_talk_to(tmp_path):
    result = satellite("--dry-run", "--no-service", env={"HOME": str(tmp_path)})
    assert result.returncode != 0
    assert "--url" in result.stderr


def test_a_dry_run_changes_nothing(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = satellite(
        "--dry-run", "--no-service", "--url", "http://example.invalid:8713",
        "--token", "s3cr3t", env={"HOME": str(home)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "would write" in result.stdout
    assert not list(home.rglob("gustav_satellite.py"))
    assert "s3cr3t" not in result.stdout


def test_it_installs_the_one_file_and_a_config_only_you_can_read(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = satellite(
        "--no-service", "--url", "http://example.invalid:8713", "--token", "s3cr3t",
        "--source", str(REPO), "--name", "kitchen",
        env={"HOME": str(home), "GUSTAV_BIN": str(home / "bin")},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    program = home / ".local" / "share" / "gustav-satellite" / "gustav_satellite.py"
    assert program.is_file()
    assert (home / "bin" / "gustav-satellite").is_file()

    config = home / ".config" / "gustav-satellite.conf"
    text = config.read_text(encoding="utf-8")
    assert "url = http://example.invalid:8713" in text
    assert "token = s3cr3t" in text
    # It holds a token, so it is not readable by the rest of the machine.
    assert oct(config.stat().st_mode)[-3:] == "600"

    # And the thing it installed runs.
    ran = subprocess.run(
        [str(home / "bin" / "gustav-satellite"), "--version"],
        capture_output=True, text=True, timeout=60,
    )
    assert ran.stdout.strip().startswith("gustav-satellite")


def test_it_keeps_a_config_that_is_already_there(tmp_path):
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    config = home / ".config" / "gustav-satellite.conf"
    config.write_text("url = http://mine:8713\nthreshold = 0.07\n", encoding="utf-8")
    result = satellite(
        "--no-service", "--url", "http://example.invalid:8713", "--source", str(REPO),
        env={"HOME": str(home), "GUSTAV_BIN": str(home / "bin")},
    )
    assert result.returncode == 0
    assert "keeping your" in result.stdout
    assert "threshold = 0.07" in config.read_text(encoding="utf-8")


# --- the one-liner rescuing a stuck install -------------------------------
# Reported twice from the same server. config.yaml is the settings of the gustav in
# that directory and was also in the repository, so an upstream edit to it stopped the
# update — and the `ai update` on a machine that is stuck is, by definition, the old
# one that does not know how to get past it. This script is what you paste into a
# stuck machine, so it has to be the thing that fixes it.
def a_checkout_whose_config_is_in_the_way(tmp_path) -> Path:
    checkout = a_checkout_with_an_origin(tmp_path)
    seed, env = tmp_path / "seed", {
        **os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    # It was in the repository, so every checkout has a tracked copy…
    (seed / "config.yaml").write_text("web:\n  enabled: false\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "--quiet", "-m", "a config"],
                   check=True, env=env)
    subprocess.run(["git", "-C", str(seed), "push", "--quiet", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(checkout), "pull", "--quiet", "--ff-only"], check=True)

    # …upstream changes it…
    (seed / "config.yaml").write_text("web:\n  enabled: true\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "--quiet", "-m", "enable the web"],
                   check=True, env=env)
    subprocess.run(["git", "-C", str(seed), "push", "--quiet", "origin", "main"], check=True)

    # …and so have you.
    (checkout / "config.yaml").write_text(
        "web:\n  enabled: true\napi:\n  port: 9000   # mine\n", encoding="utf-8")
    return checkout


def test_the_one_liner_gets_past_a_config_that_was_in_the_way(tmp_path):
    checkout = a_checkout_whose_config_is_in_the_way(tmp_path)
    # The `ai update` on a stuck machine is the old one: it does nothing but succeed.
    fake_ai(checkout, "#!/bin/sh\nexit 0\n")

    result = update_sh("--dir", str(checkout), "--ref", "main", "--no-restart")
    assert result.returncode == 0, result.stdout + result.stderr

    # The checkout moved…
    assert subprocess.run(
        ["git", "-C", str(checkout), "rev-list", "--count", "HEAD..origin/main"],
        capture_output=True, text=True, check=True,
    ).stdout.strip() == "0"
    # …your settings are still yours…
    assert "port: 9000" in (checkout / "config.yaml").read_text(encoding="utf-8")
    # …and theirs is beside it, not on top of it.
    assert (checkout / "config.yaml.new").read_text(encoding="utf-8") == (
        "web:\n  enabled: true\n"
    )
    assert "Keeping your config.yaml" in result.stdout


def test_the_settings_come_back_even_when_the_update_fails(tmp_path):
    """A file set aside and never put back is the worst outcome available."""
    checkout = a_checkout_whose_config_is_in_the_way(tmp_path)
    mine = (checkout / "config.yaml").read_text(encoding="utf-8")
    fake_ai(checkout, "#!/bin/sh\nif [ \"$2\" = --help ]; then exit 0; fi\nexit 3\n")

    result = update_sh("--dir", str(checkout), "--ref", "main", "--no-restart")
    assert result.returncode != 0
    assert (checkout / "config.yaml").read_text(encoding="utf-8") == mine


def test_a_config_nobody_has_touched_is_left_alone(tmp_path):
    checkout = a_checkout_with_an_origin(tmp_path)
    fake_ai(checkout, "#!/bin/sh\nexit 0\n")
    (checkout / "config.yaml").write_text("mine: yes\n", encoding="utf-8")   # untracked

    result = update_sh("--dir", str(checkout), "--ref", "main", "--no-restart")
    assert result.returncode == 0
    assert (checkout / "config.yaml").read_text(encoding="utf-8") == "mine: yes\n"
    assert not (checkout / "config.yaml.new").exists()
    assert "Keeping your" not in result.stdout


def test_it_says_when_the_command_will_not_be_found(tmp_path):
    """Reported from a Pi: installed fine, then `gustav-satellite: command not found`.

    ~/.local/bin is on PATH on most distributions *if it existed when you logged in*,
    which it did not five seconds ago.
    """
    home = tmp_path / "home"
    home.mkdir()
    result = satellite(
        "--no-service", "--url", "http://example.invalid:8713", "--source", str(REPO),
        env={"HOME": str(home), "GUSTAV_BIN": str(home / "bin"),
             "PATH": os.environ["PATH"]},     # which does not contain home/bin
    )
    assert result.returncode == 0
    assert "is not on your PATH" in result.stderr
    assert 'export PATH="$HOME/.local/bin:$PATH"' in result.stderr
    # And the commands it suggests afterwards use a path that works right now.
    assert f"{home}/bin/gustav-satellite --say" in result.stdout


def test_it_keeps_quiet_about_the_path_when_the_path_is_fine(tmp_path):
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    result = satellite(
        "--no-service", "--url", "http://example.invalid:8713", "--source", str(REPO),
        env={"HOME": str(home), "GUSTAV_BIN": str(home / "bin"),
             "PATH": f"{home}/bin:{os.environ['PATH']}"},
    )
    assert "is not on your PATH" not in result.stderr
    assert "gustav-satellite --say" in result.stdout


def test_a_daemon_without_voice_is_told_apart_from_no_daemon(tmp_path):
    """404 means the address is right and voice is off. That is a different fix."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):          # noqa: N802 - the stdlib's spelling
            self.send_error(404, "Not Found")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        home = tmp_path / "home"
        home.mkdir()
        result = satellite(
            "--no-service", "--url", f"http://127.0.0.1:{server.server_port}",
            "--source", str(REPO),
            env={"HOME": str(home), "GUSTAV_BIN": str(home / "bin")},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result.returncode == 0, "a server that is not ready is not a failed install"
    assert "the daemon is there, but it has no /voice" in result.stdout
    assert "voice.enabled" in result.stdout
    assert "could not reach it" not in result.stdout


def test_it_finds_a_microphone_that_is_not_the_default(tmp_path):
    """Reported from a Pi: the mic was card 1, arecord recorded the default, silence.

    The installer printed "capture: card 1: Camera" and then wrote no input_device at
    all, which is an install that looks perfect and hears nothing.
    """
    fake = tmp_path / "bin"
    fake.mkdir()
    # An arecord that lists card 1 and refuses to record from anything else.
    (fake / "arecord").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-l" ]; then\n'
        "  echo 'card 1: Camera [Terra HD1080PC Camera], device 0: USB Audio [USB Audio]'\n"
        "  exit 0\n"
        "fi\n"
        'case "$*" in *"plughw:1,0"*) exit 0 ;; esac\n'
        "echo 'arecord: main:830: audio open error: No such file or directory' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    (fake / "arecord").chmod(0o755)
    (fake / "aplay").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake / "aplay").chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        ["bash", str(SATELLITE), "--no-service", "--url", "http://example.invalid:8713",
         "--source", str(REPO)],
        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
        env={**os.environ, "HOME": str(home), "GUSTAV_BIN": str(home / "bin"),
             "PATH": f"{fake}:{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "the default device records nothing" in result.stdout
    assert "capture    plughw:1,0" in result.stdout
    config = (home / ".config" / "gustav-satellite.conf").read_text(encoding="utf-8")
    assert "input_device = plughw:1,0" in config, "and it is written down"


def test_a_default_that_works_is_left_alone(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    for name in ("arecord", "aplay"):
        (fake / name).write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "-l" ]; then echo "card 0: Thing [Thing], device 0: x [x]"; fi\n'
            "exit 0\n",
            encoding="utf-8",
        )
        (fake / name).chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        ["bash", str(SATELLITE), "--no-service", "--url", "http://example.invalid:8713",
         "--source", str(REPO)],
        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
        env={**os.environ, "HOME": str(home), "GUSTAV_BIN": str(home / "bin"),
             "PATH": f"{fake}:{os.environ['PATH']}"},
    )
    assert result.returncode == 0
    assert "the default device, which records" in result.stdout
    assert "input_device" not in (
        home / ".config" / "gustav-satellite.conf"
    ).read_text(encoding="utf-8")


def test_the_one_liner_says_what_it_moved(tmp_path):
    """It does the moving, then `ai update` says "already up to date" — which reads
    like nothing happened at all."""
    checkout = a_checkout_with_an_origin(tmp_path)
    seed, env = tmp_path / "seed", {
        **os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    (seed / "src").mkdir(exist_ok=True)
    (seed / "src" / "new.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "--quiet", "-m", "something new"],
                   check=True, env=env)
    subprocess.run(["git", "-C", str(seed), "push", "--quiet", "origin", "main"], check=True)
    fake_ai(checkout, "#!/bin/sh\nexit 0\n")

    result = update_sh("--dir", str(checkout), "--ref", "main", "--no-restart")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Moved " in result.stdout and "1 commit(s)" in result.stdout


def test_it_says_so_when_there_was_nothing_to_move(tmp_path):
    checkout = a_checkout_with_an_origin(tmp_path)
    fake_ai(checkout, "#!/bin/sh\nexit 0\n")
    result = update_sh("--dir", str(checkout), "--ref", "main", "--no-restart")
    assert result.returncode == 0
    assert "Already at " in result.stdout
