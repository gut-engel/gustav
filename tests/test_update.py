"""`ai update`: the same machine, a week later.

The weights, the database and the adapters are not this command's business — it moves
the *source* and the package and then gets out of the way. What is worth testing is
the part that has already gone wrong in the field: an update that looks like it worked
while the tree it installed is not the tree the commit describes.

pip is not run here. It is one subprocess call with everything decided before it, and
exercising it would mean a few minutes of network per test.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from gustav.update import Transcript, UpdateError, install_root, run_update

GIT = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
}


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=True, env={**os.environ, **GIT},
    )


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """An origin with two commits, and a clone that stopped after the first."""
    origin, seed, work = tmp_path / "origin.git", tmp_path / "seed", tmp_path / "gustav"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(origin), str(seed)], check=True)
    (seed / "pyproject.toml").write_text("[project]\nname = 'gustav'\n", encoding="utf-8")
    (seed / "src").mkdir()
    (seed / "src" / "needed.py").write_text("x = 1\n", encoding="utf-8")
    git(seed, "add", "-A")
    git(seed, "commit", "--quiet", "-m", "first")
    git(seed, "push", "--quiet", "origin", "main")
    subprocess.run(["git", "clone", "--quiet", str(origin), str(work)], check=True)

    (seed / "src" / "later.py").write_text("y = 2\n", encoding="utf-8")
    git(seed, "add", "-A")
    git(seed, "commit", "--quiet", "-m", "the thing you are missing")
    git(seed, "push", "--quiet", "origin", "main")
    return work


def update(root: Path, tmp_path: Path, **kwargs):
    kwargs.setdefault("reinstall", False)
    kwargs.setdefault("restart", False)
    transcript = Transcript(tmp_path / "update.log")
    try:
        return run_update(root, transcript=transcript, **kwargs)
    finally:
        transcript.close()


def test_check_says_what_is_new_and_touches_nothing(checkout, tmp_path):
    result = update(checkout, tmp_path, check_only=True)
    assert not result.up_to_date
    assert any("the thing you are missing" in line for line in result.behind)
    assert not (checkout / "src" / "later.py").exists(), "--check changes nothing"
    assert [step.name for step in result.steps] == ["fetch", "check"]


def test_it_fast_forwards_and_says_how_far(checkout, tmp_path):
    result = update(checkout, tmp_path)
    assert result.ok
    assert (checkout / "src" / "later.py").read_text(encoding="utf-8") == "y = 2\n"
    pull = next(step for step in result.steps if step.name == "pull")
    assert pull.status == "done" and "1 commit" in pull.detail


def test_a_second_update_has_nothing_to_do(checkout, tmp_path):
    update(checkout, tmp_path)
    again = update(checkout, tmp_path)
    assert again.up_to_date
    assert next(s for s in again.steps if s.name == "pull").status == "skipped"


def test_a_file_deleted_from_the_tree_is_put_back(checkout, tmp_path):
    """A fast-forward does not fill a hole in a file it had no reason to touch."""
    (checkout / "src" / "needed.py").unlink()
    result = update(checkout, tmp_path)
    assert (checkout / "src" / "needed.py").exists()
    restore = next(step for step in result.steps if step.name == "restore")
    assert restore.status == "done" and "needed.py" in restore.detail


def test_it_refuses_a_checkout_that_has_gone_its_own_way(checkout, tmp_path):
    (checkout / "src" / "mine.py").write_text("mine = True\n", encoding="utf-8")
    git(checkout, "add", "-A")
    git(checkout, "commit", "--quiet", "-m", "mine")

    with pytest.raises(UpdateError) as refused:
        update(checkout, tmp_path)
    assert "--reset" in str(refused.value)
    assert (checkout / "src" / "mine.py").exists(), "it refused; it did not help itself"


def test_reset_says_where_the_committed_work_went(checkout, tmp_path):
    (checkout / "src" / "mine.py").write_text("mine = True\n", encoding="utf-8")
    git(checkout, "add", "-A")
    git(checkout, "commit", "--quiet", "-m", "mine")
    was = git(checkout, "rev-parse", "--short", "HEAD").stdout.strip()

    result = update(checkout, tmp_path, reset=True)
    assert result.ok
    assert not (checkout / "src" / "mine.py").exists()
    assert (checkout / "src" / "later.py").exists(), "and it is now the published tree"

    pull = next(step for step in result.steps if step.name == "pull")
    assert was in pull.detail, "reset --hard is the step people regret: say where it went"
    assert git(checkout, "cat-file", "-e", was).returncode == 0, "and it is still reachable"


def test_reset_puts_uncommitted_work_in_a_stash(checkout, tmp_path):
    git(checkout, "commit", "--quiet", "--allow-empty", "-m", "a commit of my own")
    (checkout / "src" / "needed.py").write_text("edited by hand\n", encoding="utf-8")

    result = update(checkout, tmp_path, reset=True)
    assert result.ok
    assert git(checkout, "stash", "list").stdout.strip(), "the edit is recoverable"
    assert "stash" in next(s for s in result.steps if s.name == "pull").detail


def test_a_dry_run_moves_nothing(checkout, tmp_path):
    result = update(checkout, tmp_path, dry_run=True)
    assert not (checkout / "src" / "later.py").exists()
    assert [s.status for s in result.steps if s.name in {"pull", "install", "verify"}] == [
        "skipped", "skipped", "skipped",
    ]


def test_a_wheel_install_is_told_it_cannot_be_updated_this_way(tmp_path):
    plain = tmp_path / "not-a-checkout"
    plain.mkdir()
    (plain / "pyproject.toml").write_text("[project]\nname = 'gustav'\n", encoding="utf-8")
    with pytest.raises(UpdateError) as refused:
        update(plain, tmp_path)
    assert "not a git checkout" in str(refused.value)


def test_everything_it_ran_is_in_the_log(checkout, tmp_path):
    update(checkout, tmp_path)
    written = (tmp_path / "update.log").read_text(encoding="utf-8")
    assert "$ git -C" in written and "fetch" in written
    assert "-> exit 0" in written
    assert str(checkout) in written


def test_it_updates_the_checkout_this_ai_runs_from(tmp_path):
    """No argument means this install, not whatever directory you are standing in."""
    root = install_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "gustav" / "update.py").exists()
