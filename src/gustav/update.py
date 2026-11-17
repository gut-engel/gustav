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
  interesting failures here happen inside pip and git and scroll past. The log is
  written for the question that actually gets asked a week later — *what changed, and
  was this machine any different from the one it worked on* — so it opens with the
  machine, the interpreter, the remote and the room on disk; it names the commit it
  was on and the commit it ended on, with the diffstat between them; it keeps pip's
  whole verbose attempt, then what pip thought of the result; and it times every step.
  The last ten are kept.
"""

from __future__ import annotations

import getpass
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gustav import service


class UpdateError(RuntimeError):
    """Something the person has to decide about, said plainly."""


#: Files in the checkout that belong to whoever runs it, not to the project.
#:
#: ``config.yaml`` is the settings of the gustav that lives in this directory — it is
#: edited on every machine, by definition — and it was also in the repository as a
#: template. So the day an upstream commit touched it, every update everywhere stopped
#: dead with "your local changes would be overwritten by merge". It is no longer
#: tracked, and this list is what carries the ones that still are across the pull.
YOURS = ("config.yaml",)


def _yours(porcelain: list[str]) -> list[str]:
    """The dirty files from `git status --porcelain` that are the person's."""
    found = []
    for line in porcelain:
        path = line[3:].strip() if len(line) > 3 else ""
        if " -> " in path:          # a rename; not something to second-guess
            continue
        if path in YOURS:
            found.append(path)
    return found


# --- where the log goes ------------------------------------------------------


def log_directory() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / "gustav"


class Transcript:
    """Every command and everything it said, in a file, with timestamps."""

    def __init__(self, path: Path | None = None) -> None:
        pruned = path is None
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
        if pruned:
            # Only when this transcript chose the path itself: a log someone asked for
            # by name is theirs, and so is the directory it is in.
            prune_logs(directory=path.parent)
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
        """One command, everything it said, and how it ended.

        Masked on the way in, not on the way out: a remote URL with a token in it
        (``https://x-access-token:…@github.com``) is printed by ``git remote -v`` and
        by half of pip's output, and this file gets pasted into issues.
        """
        self.write(f"$ {_mask(' '.join(argv))}")
        if self._handle is None:
            return
        for stream in (done.stdout, done.stderr):
            if stream:
                text = _mask(stream)
                self._handle.write(text if text.endswith("\n") else text + "\n")
        self._handle.write(f"  -> exit {done.returncode}\n")
        self._handle.flush()

    def write_raw(self, line: str) -> None:
        """A line of somebody else's output, as it was written, flushed immediately."""
        if self._handle is None:
            return
        self._handle.write(_mask(line if line.endswith("\n") else line + "\n"))
        self._handle.flush()

    def section(self, title: str) -> None:
        """A heading, so a log read in a hurry can be skimmed."""
        self.write(f"--- {title} " + "-" * max(4, 60 - len(title)))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def prune_logs(keep: int = 10, directory: Path | None = None) -> list[Path]:
    """Keep the last few update logs. One nobody can find and nobody deletes is a leak."""
    directory = directory or log_directory()
    try:
        found = sorted(directory.glob("update-*.log"), key=lambda p: p.name, reverse=True)
    except OSError:
        return []
    removed = []
    for old in found[keep:]:
        try:
            old.unlink()
            removed.append(old)
        except OSError:
            pass
    return removed


# --- the steps ---------------------------------------------------------------


@dataclass
class Step:
    """One thing that was done, and how it went."""

    name: str
    status: str  # done | skipped | failed
    detail: str = ""
    #: Wall-clock seconds. The one number that answers "what is it doing?" afterwards.
    seconds: float = 0.0

    @property
    def took(self) -> str:
        return f"{self.seconds:.1f}s" if self.seconds >= 0.05 else ""

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
    #: The commit it was on, and the one it ended on. Short shas, for the screen.
    was: str = ""
    now: str = ""
    #: What moved, as git counts it: files, insertions, deletions.
    changed: str = ""
    #: Packages in the environment afterwards, and seconds for the whole thing.
    packages: int = 0
    seconds: float = 0.0

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


def _version() -> str:
    from gustav import __version__

    return __version__


def _os_name() -> str:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.platform()


def _mask(text: str) -> str:
    """Credentials live in remote URLs and index URLs, and logs get pasted into issues."""
    return re.sub(r"://[^/@\s]*:[^/@\s]*@", "://***@", text or "")


def _machine(root: Path, transcript: Transcript) -> None:
    """What this machine is, before anything about it changes.

    The question a week later is never "what did the command print" — it is "was this
    machine different from the one it worked on". None of this is expensive and all of
    it has, at some point, been the answer.
    """
    transcript.section("the machine")
    usage = shutil.disk_usage(root if root.exists() else root.parent)
    for line in (
        f"gustav    {_version()}",
        f"user      {getpass.getuser()} (uid {os.getuid()})",
        f"machine   {platform.node()} — {platform.system()} {platform.release()} "
        f"{platform.machine()}",
        f"os        {_os_name()}",
        f"python    {platform.python_version()} at {sys.executable}",
        f"root      {root}",
        f"disk      {usage.free / (1 << 30):.1f} GiB free of {usage.total / (1 << 30):.1f} GiB",
    ):
        transcript.write(f"  {line}")

    # The settings that decide what pip will do, which is the step that goes wrong.
    interesting = ("PIP_", "PYTHON", "VIRTUAL_ENV", "CUDA", "LD_LIBRARY_PATH",
                   "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy")
    env = sorted(
        f"  {key}={_mask(value)}"
        for key, value in os.environ.items()
        if key.startswith(interesting)
    )
    transcript.write("  environment pip will see:")
    transcript.write("\n".join(env) if env else "    (nothing set)")
    for argv in ([sys.executable, "-VV"], [sys.executable, "-m", "pip", "--version"]):
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        transcript.command(argv, done)


#: Lines from pip worth repeating to someone watching. pip -v says everything it does
#: — which is what the log is for and not what a one-line status is for.
_WORTH_SAYING = re.compile(
    r"^\s*(Collecting|Downloading|Using cached|Obtaining|Building|Preparing|Processing"
    r"|Installing|Successfully|Requirement already|Uninstalling|error|ERROR|WARNING)",
)


def _run_watched(
    argv: list[str],
    transcript: Transcript,
    progress: Callable[[str], None] | None,
    *,
    what: str = "",
    timeout: float = 3600.0,
) -> subprocess.CompletedProcess:
    """Run something slow, writing its output down *as it happens*.

    Two reasons, and the second is the one that was reported. The log becomes live, so
    `tail -f` on it during an update shows something. And the caller gets each line as
    it arrives, which is how a command that spends four minutes inside pip can say so
    instead of looking identical to a hang.
    """
    transcript.write(f"$ {_mask(' '.join(argv))}")
    started = time.monotonic()
    collected: list[str] = []
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_said = 0.0
    pending = ""

    def shown(line: str, at: float) -> str:
        return (f"[{int(at - started) // 60}:{int(at - started) % 60:02d}] "
                f"{what + ': ' if what else ''}{line.strip()[:70]}")

    try:
        for line in process.stdout:          # type: ignore[union-attr]
            collected.append(line)
            transcript.write_raw(line)
            if progress is None:
                continue
            now = time.monotonic()
            # Not every line: pip -v produces thousands, and a status that scrolls is
            # no more readable than silence. One worth repeating, at most twice a
            # second — and whatever was suppressed last is shown at the end, so the
            # line left on the screen is always the most recent thing that happened.
            if not _WORTH_SAYING.match(line):
                continue
            if now - last_said > 0.5:
                last_said = now
                pending = ""
                progress(shown(line, now))
            else:
                pending = line
        if progress is not None and pending:
            progress(shown(pending, time.monotonic()))
    finally:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    output = "".join(collected)
    transcript.write(f"  -> exit {process.returncode}")
    return subprocess.CompletedProcess(argv, process.returncode, output, "")


def _git(root: Path, *args: str, transcript: Transcript | None = None) -> subprocess.CompletedProcess:
    argv = ["git", "-C", str(root), *args]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if transcript is not None:
        # Logged without the -C: the root is named once, in the header, and repeating
        # an absolute path on every line is what makes a log unreadable.
        transcript.command(["git", *args], done)
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
    progress: Callable[[str], None] | None = None,
) -> Update:
    """Fetch, fast-forward, reinstall, verify, restart. In that order, and no further.

    ``progress`` is handed each line worth repeating from the slow step, so a caller
    with a terminal can show that pip is working rather than leaving four minutes of
    nothing on the screen.
    """
    root = install_root(root)
    own = transcript is None
    transcript = transcript or Transcript()
    try:
        return _run(
            root, ref, check_only, reset, reinstall, restart, dry_run, transcript,
            progress,
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
    progress: Callable[[str], None] | None = None,
) -> Update:
    started = time.monotonic()
    transcript.write(f"ai update {_version()} — {root}")
    transcript.write(
        f"  options: ref={ref or '<current branch>'} check_only={check_only} "
        f"reset={reset} reinstall={reinstall} restart={restart} dry_run={dry_run}"
    )
    _machine(root, transcript)
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

    transcript.section("where it is starting from")
    state.was = _git(root, "rev-parse", "--short", "HEAD", transcript=transcript).stdout.strip()
    _git(root, "log", "-1", "--format=%h %ad %an — %s", "--date=short", transcript=transcript)
    _git(root, "remote", "-v", transcript=transcript)
    _git(root, "status", "--short", "--branch", transcript=transcript)

    transcript.section("fetch")
    at = time.monotonic()
    fetched = _git(root, "fetch", "--quiet", "origin", ref, transcript=transcript)
    took = time.monotonic() - at
    if fetched.returncode != 0:
        state.steps.append(Step("fetch", "failed", (fetched.stderr or "").strip(), took))
        return state
    state.steps.append(Step("fetch", "done", f"origin/{ref}", took))

    log = _git(root, "log", "--oneline", f"HEAD..origin/{ref}", transcript=transcript)
    state.behind = [line for line in log.stdout.splitlines() if line.strip()]
    if state.behind:
        # Not just the subjects: who wrote them and when, and which files they touch.
        # This is the half of the log anyone actually reads afterwards.
        _git(root, "log", "--format=%h %ad %an — %s", "--date=short",
             f"HEAD..origin/{ref}", transcript=transcript)
        stat = _git(root, "diff", "--stat", f"HEAD..origin/{ref}", transcript=transcript)
        tail = [line for line in stat.stdout.splitlines() if line.strip()]
        state.changed = tail[-1].strip() if tail else ""

    if check_only:
        state.now = state.was
        state.seconds = time.monotonic() - started
        state.steps.append(
            Step("check", "done", "already up to date" if state.up_to_date
                 else f"{len(state.behind)} new commit(s)"
                      + (f", {state.changed}" if state.changed else ""))
        )
        transcript.write(
            f"finished: --check only, nothing was changed, in {state.seconds:.1f}s"
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
            + ("  (your config.yaml is not the problem; that one is kept either way)\n"
               if _yours(local) else "")
            + (f"  {commits_ahead} commit(s) of its own\n" if commits_ahead else "")
            + ("".join(f"  {line}\n" for line in local) if local else "")
            + f"Pass --reset to match origin/{ref}: uncommitted work goes to a git stash,\n"
            + "and it tells you the commit this was on so nothing becomes unreachable.\n"
            + "Or sort it out by hand and run it again."
        )

    # Whatever is the person's is carried over the pull rather than standing in its
    # way. Read first, then put the tracked copy back so git has nothing to object to.
    yours: dict[str, str] = {}
    if not dry_run:
        for path in _yours(local):
            try:
                yours[path] = (root / path).read_text(encoding="utf-8")
            except OSError as exc:
                transcript.write(f"could not read {path}, leaving it alone: {exc}")
        if yours:
            transcript.section("setting your own files aside")
            for path in yours:
                _git(root, "checkout", "--", path, transcript=transcript)

    transcript.section("pull")
    at = time.monotonic()
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
        state.steps.append(
            Step("pull", "done",
                 f"{len(state.behind)} commit(s)" + (f", {state.changed}" if state.changed else ""),
                 time.monotonic() - at)
        )

    if yours:
        transcript.section("putting your own files back")
        restored, fresh = [], []
        for path, text in yours.items():
            target = root / path
            # If the commit that just arrived has its own version, it is kept *beside*
            # yours rather than over it: nobody wants their settings replaced, and
            # nobody wants to be told nothing changed either.
            incoming = None
            try:
                incoming = target.read_text(encoding="utf-8")
            except OSError:
                pass
            if incoming is not None and incoming != text:
                (root / f"{path}.new").write_text(incoming, encoding="utf-8")
                fresh.append(f"{path}.new")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            restored.append(path)
            transcript.write(f"kept your {path}" + (f"; theirs is in {path}.new" if fresh else ""))
        detail = f"kept your {', '.join(sorted(restored))}"
        if fresh:
            detail += f"; the version from the update is in {', '.join(sorted(fresh))}"
        state.steps.append(Step("yours", "done", detail))

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
        transcript.section("install")
        argv = [
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--progress-bar", "off", "-v", "-e", f"{root}{_extras()}",
        ]
        at = time.monotonic()
        done = _run_watched(argv, transcript, progress, what="pip")
        took = time.monotonic() - at
        if done.returncode != 0:
            said = (done.stderr or done.stdout or "").strip().splitlines()
            state.steps.append(
                Step("install", "failed", said[-1] if said else "pip failed", took)
            )
            return state
        # "It worked on mine" is answerable from these two and nothing else.
        transcript.section("what the environment holds now")
        listed = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=300,
        )
        transcript.command([sys.executable, "-m", "pip", "list", "--format=freeze"], listed)
        state.packages = len([x for x in listed.stdout.splitlines() if x.strip()])
        checked = subprocess.run(
            [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=300
        )
        transcript.command([sys.executable, "-m", "pip", "check"], checked)
        detail = f"pip install -e {root}{_extras()}"
        if state.packages:
            detail += f"; {state.packages} packages"
        if checked.returncode != 0:
            detail += "; pip check is unhappy (the log says why)"
        state.steps.append(Step("install", "done", detail, took))

    if dry_run:
        state.steps.append(Step("verify", "skipped", "would byte-compile the source"))
    else:
        transcript.section("verify")
        argv = [sys.executable, "-m", "compileall", "-q", str(root / "src")]
        at = time.monotonic()
        done = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        transcript.command(argv, done)
        if done.returncode != 0:
            state.steps.append(
                Step("verify", "failed", "some of src/ does not compile",
                     time.monotonic() - at)
            )
            return state
        files = len(list((root / "src").rglob("*.py")))
        state.steps.append(
            Step("verify", "done", f"{files} source files compile", time.monotonic() - at)
        )

    if not restart:
        state.steps.append(Step("restart", "skipped", "not asked for"))
    elif dry_run:
        state.steps.append(Step("restart", "skipped", "would restart gustav.service if it is up"))
    else:
        transcript.section("restart")
        at = time.monotonic()
        step = _restart(transcript)
        step.seconds = time.monotonic() - at
        state.steps.append(step)
        state.restarted = step.status == "done"

    transcript.section("where it ended up")
    state.now = _git(root, "rev-parse", "--short", "HEAD", transcript=transcript).stdout.strip()
    _git(root, "log", "-1", "--format=%h %ad %an — %s", "--date=short", transcript=transcript)
    _git(root, "status", "--short", "--branch", transcript=transcript)
    state.seconds = time.monotonic() - started
    transcript.write(
        f"finished: {state.was or '?'} -> {state.now or '?'}"
        + (f" ({state.changed})" if state.changed else "")
        + f" in {state.seconds:.1f}s"
    )
    for step in state.steps:
        transcript.write(f"  {step.mark} {step.name:<9}{step.detail}"
                         + (f"  [{step.took}]" if step.took else ""))
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
