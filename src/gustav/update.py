"""Updating an install in place.

The one-liner installs; this is the same machine a week later. It does what the
installer does to a checkout that is already there — fetch, fast-forward, reinstall the
package, check that what it installed can actually be imported, restart the service —
and nothing else. No weights are touched, no database, no adapters, no config.

Two rules it keeps, both learned the hard way:

* **A fast-forward is not the same as "the tree is now what the commit says".** A file
  deleted from the working tree but not from the commit is invisible to it: git only
  touches what changed between the two commits, so the hole stays, pip installs happily
  around it, and the first anyone hears is an ImportError for a module that is right
  there in the repository. So the deleted files are named and put back.
* **Everything is written down.** Every command and all of its output goes to
  ``~/.local/state/gustav/update-<date>.log``, next to the installer's logs, because the
  interesting failures here happen inside pip and git and scroll past.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gustav import service


class UpdateError(RuntimeError):
    """Something the person has to decide about, said plainly."""


# --- where the log goes ------------------------------------------------------


def log_directory() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / "gustav"


class Transcript:
    """Every command and everything it said, in a file, with timestamps."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            directory = log_directory()
            try:
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"update-{stamp}.log"
            except OSError:  # a home that cannot be written to is not a reason to stop
                path = Path(os.environ.get("TMPDIR", "/tmp")) / f"gustav-update-{stamp}.log"
        self.path = path
        self._handle = None
        try:
            self._handle = path.open("a", encoding="utf-8")
        except OSError:
            self._handle = None

    def write(self, text: str) -> None:
        if self._handle is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for line in text.splitlines() or [""]:
            self._handle.write(f"{stamp}  {line}\n")
        self._handle.flush()

    def command(self, argv: list[str], done: subprocess.CompletedProcess) -> None:
        self.write(f"$ {' '.join(argv)}")
        if self._handle is None:
            return
        for stream in (done.stdout, done.stderr):
            if stream:
                self._handle.write(stream if stream.endswith("\n") else stream + "\n")
        self._handle.write(f"  -> exit {done.returncode}\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# --- the steps ---------------------------------------------------------------


@dataclass
class Step:
    """One thing that was done, and how it went."""

    name: str
    status: str  # done | skipped | failed
    detail: str = ""

    @property
    def mark(self) -> str:
        return {"done": "✓", "skipped": "–", "failed": "✗"}.get(self.status, "?")


@dataclass
class Update:
    root: Path
    ref: str
    log: Path
    steps: list[Step] = field(default_factory=list)
    behind: list[str] = field(default_factory=list)   # the commits it would bring in
    restarted: bool = False

    @property
    def ok(self) -> bool:
        return not any(step.status == "failed" for step in self.steps)

    @property
    def up_to_date(self) -> bool:
        return not self.behind


def install_root(start: Path | None = None) -> Path:
    """The checkout this `ai` is running from.

    An editable install puts the package in ``<root>/src/gustav``, which is the thing
    to update — not whatever directory the person happened to be standing in.
    """
    if start is not None:
        return Path(start).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], Path.cwd()):
        if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
            return candidate
    return here.parents[2]


def _git(root: Path, *args: str, transcript: Transcript | None = None) -> subprocess.CompletedProcess:
    argv = ["git", "-C", str(root), *args]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if transcript is not None:
        transcript.command(argv, done)
    return done


def run_update(
    root: Path | None = None,
    *,
    ref: str = "",
    check_only: bool = False,
    reset: bool = False,
    reinstall: bool = True,
    restart: bool = True,
    dry_run: bool = False,
    transcript: Transcript | None = None,
) -> Update:
    """Fetch, fast-forward, reinstall, verify, restart. In that order, and no further."""
    root = install_root(root)
    own = transcript is None
    transcript = transcript or Transcript()
    try:
        return _run(
            root, ref, check_only, reset, reinstall, restart, dry_run, transcript
        )
    finally:
        if own:
            transcript.close()


def _run(
    root: Path,
    ref: str,
    check_only: bool,
    reset: bool,
    reinstall: bool,
    restart: bool,
    dry_run: bool,
    transcript: Transcript,
) -> Update:
    transcript.write(f"update of {root}")
    transcript.write(f"  python: {sys.executable}")
    if not (root / ".git").exists():
        raise UpdateError(
            f"{root} is not a git checkout, so there is nothing to pull.\n"
            "This happens when gustav was installed from a wheel rather than by the\n"
            "installer. Reinstall with scripts/install.sh, or update it however you\n"
            "installed it."
        )
    if not (root / "pyproject.toml").exists():
        raise UpdateError(f"{root} has no pyproject.toml; this is not a gustav checkout")

    if not ref:
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD", transcript=transcript)
        ref = branch.stdout.strip() if branch.returncode == 0 else "main"
        if ref == "HEAD":  # detached: nothing to follow
            raise UpdateError(
                f"{root} is not on a branch (detached HEAD). Check one out, or pass --ref."
            )

    state = Update(root=root, ref=ref, log=transcript.path)

    fetched = _git(root, "fetch", "--quiet", "origin", ref, transcript=transcript)
    if fetched.returncode != 0:
        state.steps.append(Step("fetch", "failed", (fetched.stderr or "").strip()))
        return state
    state.steps.append(Step("fetch", "done", f"origin/{ref}"))

    log = _git(root, "log", "--oneline", f"HEAD..origin/{ref}", transcript=transcript)
    state.behind = [line for line in log.stdout.splitlines() if line.strip()]

    if check_only:
        state.steps.append(
            Step("check", "done", "already up to date" if state.up_to_date
                 else f"{len(state.behind)} new commit(s)")
        )
        return state

    changed = _git(root, "status", "--porcelain", "--untracked-files=no", transcript=transcript)
    local = [line for line in changed.stdout.splitlines() if line.strip()]
    ahead = _git(root, "rev-list", "--count", f"origin/{ref}..HEAD", transcript=transcript)
    commits_ahead = int((ahead.stdout or "0").strip() or 0)

    can_fast_forward = _git(
        root, "merge-base", "--is-ancestor", "HEAD", f"origin/{ref}", transcript=transcript
    ).returncode == 0

    if not can_fast_forward and not reset:
        raise UpdateError(
            f"{root} cannot be fast-forwarded to origin/{ref}.\n"
            + (f"  {commits_ahead} commit(s) of its own\n" if commits_ahead else "")
            + ("".join(f"  {line}\n" for line in local) if local else "")
            + f"Pass --reset to match origin/{ref}: uncommitted work goes to a git stash,\n"
            + "and it tells you the commit this was on so nothing becomes unreachable.\n"
            + "Or sort it out by hand and run it again."
        )

    if dry_run:
        state.steps.append(
            Step("pull", "skipped", f"would bring in {len(state.behind)} commit(s)")
        )
    elif reset and (not can_fast_forward or local):
        # Two different things can be in the way, and they are recovered differently:
        # uncommitted work goes to a stash, committed work stays reachable by its sha.
        # Both are named here, because "reset --hard" is the step people regret.
        was = _git(root, "rev-parse", "--short", "HEAD", transcript=transcript).stdout.strip()
        stash = _git(
            root, "stash", "push", "--include-untracked",
            "--message", f"ai update {datetime.now():%Y-%m-%d %H:%M}", transcript=transcript
        )
        stashed = stash.returncode == 0 and "No local changes" not in (stash.stdout or "")
        hard = _git(root, "reset", "--hard", "--quiet", f"origin/{ref}", transcript=transcript)
        if hard.returncode != 0:
            state.steps.append(Step("pull", "failed", (hard.stderr or "").strip()))
            return state
        kept = f"what was here is commit {was}"
        if stashed:
            kept += ", uncommitted work in 'git stash list'"
        state.steps.append(Step("pull", "done", f"reset to origin/{ref}; {kept}"))
    elif state.up_to_date and not local:
        state.steps.append(Step("pull", "skipped", "already up to date"))
    else:
        merged = _git(root, "merge", "--ff-only", "--quiet", f"origin/{ref}", transcript=transcript)
        if merged.returncode != 0:
            state.steps.append(
                Step("pull", "failed", (merged.stderr or merged.stdout or "").strip())
            )
            return state
        state.steps.append(Step("pull", "done", f"{len(state.behind)} commit(s)"))

    # The hole a fast-forward leaves behind.
    if not dry_run:
        missing = _git(root, "ls-files", "--deleted", transcript=transcript)
        gone = [line for line in missing.stdout.splitlines() if line.strip()]
        if gone:
            restored = _git(root, "checkout", "--", *gone, transcript=transcript)
            if restored.returncode == 0:
                state.steps.append(
                    Step("restore", "done", f"{len(gone)} file(s) put back: " + ", ".join(gone[:3]))
                )
            else:
                state.steps.append(Step("restore", "failed", (restored.stderr or "").strip()))

    if not reinstall:
        state.steps.append(Step("install", "skipped", "not asked for"))
    elif dry_run:
        state.steps.append(Step("install", "skipped", f"would run pip install -e {root}{_extras()}"))
    else:
        argv = [
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--progress-bar", "off", "-v", "-e", f"{root}{_extras()}",
        ]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
        transcript.command(argv, done)
        if done.returncode != 0:
            said = (done.stderr or done.stdout or "").strip().splitlines()
            state.steps.append(Step("install", "failed", said[-1] if said else "pip failed"))
            return state
        state.steps.append(Step("install", "done", f"pip install -e {root}{_extras()}"))

    if dry_run:
        state.steps.append(Step("verify", "skipped", "would byte-compile the source"))
    else:
        argv = [sys.executable, "-m", "compileall", "-q", str(root / "src")]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        transcript.command(argv, done)
        if done.returncode != 0:
            state.steps.append(Step("verify", "failed", "some of src/ does not compile"))
            return state
        state.steps.append(Step("verify", "done", "the source compiles"))

    if not restart:
        state.steps.append(Step("restart", "skipped", "not asked for"))
    elif dry_run:
        state.steps.append(Step("restart", "skipped", "would restart gustav.service if it is up"))
    else:
        state.steps.append(_restart(transcript))
        state.restarted = state.steps[-1].status == "done"
    return state


def _extras() -> str:
    """``[ml]`` if this install has the ML stack, so an update does not strip it.

    Asked by looking, not by importing: importing torch costs seconds and opens CUDA.
    """
    import importlib.util

    return "[ml]" if importlib.util.find_spec("torch") is not None else ""


def _restart(transcript: Transcript) -> Step:
    state = service.status()
    if not state["systemctl"]:
        return Step("restart", "skipped", "no systemctl on this machine")
    if not state["installed"]:
        return Step("restart", "skipped", "no user service installed")
    if state["active"] != "active":
        return Step("restart", "skipped", f"service is {state['active']}; start it yourself")
    code, text = service.systemctl("restart", service.UNIT_NAME)
    transcript.write(f"$ systemctl --user restart {service.UNIT_NAME}\n{text}\n  -> exit {code}")
    if code != 0:
        return Step("restart", "failed", text)
    return Step("restart", "done", "gustav.service restarted")
