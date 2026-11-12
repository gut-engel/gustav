"""What a tool is allowed to do, and how that is enforced.

A tool is source code the *model* wrote, running on your machine. That is a different
risk from a practice exercise: an exercise is thrown away after its tests run, while a
tool is kept, called again, and called with arguments that came from a conversation. So
the limits are declared per tool, checked before review, and enforced at runtime.

Enforcement is four things at once, because none of them is sufficient alone:

1. **A separate process**, with ``RLIMIT_AS``, ``RLIMIT_CPU``, ``RLIMIT_FSIZE`` and
   ``RLIMIT_NPROC``, a wall-clock timeout and ``SIGKILL`` to the whole session. An
   infinite loop or an allocation storm costs seconds and nothing else.
2. **A module allowlist, checked statically, and a denylist at runtime.** The tool's
   own imports are checked against ``tools.allowed_modules`` before review, where a
   refusal can be explained. At runtime the audit hook enforces a smaller *denylist*
   instead — ``socket``, ``subprocess``, ``ctypes`` and friends — because an allowlist
   at import time would also catch the stdlib importing itself: ``pathlib`` imports
   ``os``, and a tool that may use ``pathlib`` has to be allowed to.
3. **A filesystem allowlist**, enforced by an audit hook on ``open``: writes only
   inside the tool's own throwaway workspace, reads only there and under the paths you
   listed in ``tools.readable_paths``. The Python installation itself is readable
   because it has to be — importing a module the interpreter has not cached yet *is* a
   file read, and a tool that may use ``datetime`` has to be allowed to load it.
4. **Hard refusals** for ``subprocess``, ``os.system``, ``os.exec*``, ``os.fork`` and
   socket use, on the audit events rather than by patching names — a hook cannot be
   removed once installed, while a patched attribute can be put back.

**What this does not guarantee.** CPython's audit hooks are a monitoring interface, not
a security boundary: code that can reach a C extension or an interpreter bug can get
around them, and a tool that spends its ten seconds doing arithmetic can still return a
wrong answer. The real boundaries are the ones a human provides — *you* read the source
before it can be called, and the capability it declared is in front of you when you do.
If you intend to run tools you have not read, put a container or a seccomp filter under
``ToolRunner``, which is why everything here goes through one subprocess call.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

#: Every capability a tool may declare, and what it means in one line.
CAPABILITIES: dict[str, str] = {
    "pure": "compute only: no files, no network, no subprocesses",
    "write_workspace": "read and write files inside its own throwaway directory",
    "read_paths": "read files under the directories you listed in tools.readable_paths",
    "network": "HTTP requests to the hosts you listed in tools.allowed_hosts",
}

#: Modules a tool's own source may never mention, whatever the allowlist says. Refused
#: at review time with a reason, rather than silently filtered.
NEVER_IMPORTABLE = frozenset(
    {
        "ctypes", "subprocess", "socket", "multiprocessing", "threading", "signal",
        "importlib", "builtins", "gc", "inspect", "marshal", "pickle", "shutil",
        "sys", "os", "pty", "fcntl", "mmap", "resource", "runpy", "site",
        "sysconfig", "tempfile", "webbrowser", "http", "ftplib", "smtplib",
        "telnetlib", "xmlrpc", "asyncio", "ssl", "code", "codeop", "compileall",
    }
)

#: Modules that may not be imported *at runtime*, however they are reached — including
#: through ``__import__`` in a string the model built at call time. Deliberately
#: smaller than the list above: these are the ones no part of the permitted stdlib
#: needs, which is checked by a test, so blocking them breaks nothing legitimate.
BLOCKED_AT_RUNTIME = frozenset(
    {
        "ctypes", "socket", "subprocess", "multiprocessing", "asyncio", "ssl",
        "pty", "mmap", "telnetlib", "ftplib", "smtplib", "webbrowser", "xmlrpc",
        "http", "urllib.request", "socketserver", "http.client", "http.server",
    }
)


@dataclass
class Policy:
    """The limits one tool runs under."""

    capabilities: tuple[str, ...] = ("pure",)
    allowed_modules: tuple[str, ...] = ()
    readable_paths: tuple[Path, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    workspace: Path | None = None
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 512
    max_output_bytes: int = 64 * 1024
    notes: list[str] = field(default_factory=list)

    @property
    def may_write(self) -> bool:
        return "write_workspace" in self.capabilities

    @property
    def may_read_paths(self) -> bool:
        return "read_paths" in self.capabilities

    @property
    def may_network(self) -> bool:
        return "network" in self.capabilities and bool(self.allowed_hosts)

    def describe(self) -> list[str]:
        lines = [f"{name}: {CAPABILITIES.get(name, 'unknown capability')}"
                 for name in self.capabilities]
        if self.may_read_paths:
            lines.append("readable: " + ", ".join(str(p) for p in self.readable_paths))
        if self.may_network:
            lines.append("hosts: " + ", ".join(self.allowed_hosts))
        lines.append(
            f"limits: {self.timeout_seconds:.0f}s, {self.memory_limit_mb} MiB, "
            f"{len(self.allowed_modules)} importable modules"
        )
        return lines

    def to_dict(self) -> dict[str, object]:
        return {
            "capabilities": list(self.capabilities),
            "allowed_modules": list(self.allowed_modules),
            "blocked_modules": sorted(BLOCKED_AT_RUNTIME),
            "readable_paths": [str(p) for p in self.readable_paths],
            "allowed_hosts": list(self.allowed_hosts),
            "workspace": str(self.workspace) if self.workspace else None,
            "timeout_seconds": self.timeout_seconds,
            "memory_limit_mb": self.memory_limit_mb,
        }


def policy_for(
    capabilities: list[str] | tuple[str, ...],
    config,
    *,
    workspace: Path | None = None,
) -> Policy:
    """The policy a tool with these capabilities runs under, from your configuration."""
    wanted = tuple(dict.fromkeys(c.strip() for c in capabilities if c.strip())) or ("pure",)
    return Policy(
        capabilities=wanted,
        allowed_modules=tuple(config.allowed_modules),
        readable_paths=tuple(Path(p).expanduser().resolve() for p in config.readable_paths),
        allowed_hosts=tuple(config.allowed_hosts),
        workspace=workspace,
        timeout_seconds=config.timeout_seconds,
        memory_limit_mb=config.memory_limit_mb,
        max_output_bytes=config.max_output_bytes,
    )


def unsupported(capabilities: list[str] | tuple[str, ...], allowed: list[str]) -> list[str]:
    """Capabilities that are either unknown or more than you allow."""
    permitted = set(allowed)
    return [
        c
        for c in capabilities
        if c not in CAPABILITIES or c not in permitted
    ]


#: The guard, as source, because it has to run inside the tool's own process.
#: ``__POLICY__`` is replaced with a JSON blob — a plain substitution rather than
#: ``str.format``, because this source is full of braces. It *defines* the hook and
#: ``_install_guard()``; the harness decides when to arm it.
GUARD = '''\
import json as _json, sys as _sys, resource as _resource, posixpath as _posixpath
_POLICY = _json.loads(__POLICY__)
_BLOCKED = set(_POLICY["blocked_modules"])
_READABLE = list(_POLICY["readable_paths"])
_WORKSPACE = _POLICY["workspace"]
_RUNTIME = list(_POLICY["runtime_paths"])
_MAY_WRITE = "write_workspace" in _POLICY["capabilities"]
_MAY_READ = "read_paths" in _POLICY["capabilities"]
_MAY_NET = "network" in _POLICY["capabilities"] and bool(_POLICY["allowed_hosts"])
_HOSTS = set(_POLICY["allowed_hosts"])


class ToolRefused(Exception):
    """The tool asked for something its capabilities do not allow."""


def _apply_limits():
    for _name, _value in (
        ("RLIMIT_AS", _POLICY["memory_limit_mb"] * 1024 * 1024),
        ("RLIMIT_DATA", _POLICY["memory_limit_mb"] * 1024 * 1024),
        ("RLIMIT_CPU", int(_POLICY["timeout_seconds"]) + 1),
        ("RLIMIT_FSIZE", 16 * 1024 * 1024),
        ("RLIMIT_NPROC", 32),
        ("RLIMIT_NOFILE", 64),
    ):
        _limit = getattr(_resource, _name, None)
        if _limit is None or not _value:
            continue
        try:
            # The hard limit too, so the tool cannot raise it again.
            _resource.setrlimit(_limit, (_value, _value))
        except (ValueError, OSError):
            pass


def _module_blocked(name):
    """Is this import one of the few that is never allowed?

    Matched on the dotted prefix, so blocking "http" blocks "http.client" too, and the
    tool cannot reach a blocked module by importing something under it.
    """
    if not name:
        return False
    parts = str(name).split(".")
    for index in range(1, len(parts) + 1):
        if ".".join(parts[:index]) in _BLOCKED:
            return True
    return False


def _under(resolved, roots):
    return any(resolved == root or resolved.startswith(root + "/") for root in roots)


def _path_allowed(path, writing):
    try:
        resolved = _posixpath.realpath(str(path))
    except Exception:
        return False
    if _WORKSPACE and (resolved == _WORKSPACE or resolved.startswith(_WORKSPACE + "/")):
        return _MAY_WRITE or not writing
    if writing:
        return False
    # The interpreter's own files: importing a module reads one, so refusing these
    # would mean refusing `import datetime`. Read-only, and never the project.
    if _under(resolved, _RUNTIME):
        return True
    if not _MAY_READ:
        return False
    return _under(resolved, _READABLE)


_WRITE_FLAGS = ("w", "a", "x", "+")


def _hook(event, args):
    if event == "import":
        name = args[0] if args else ""
        if _module_blocked(name):
            raise ToolRefused("this tool may not import " + repr(name))
        return
    if event == "open":
        path = args[0] if args else None
        mode = args[1] if len(args) > 1 else ""
        if path is None:
            return
        writing = any(flag in str(mode or "") for flag in _WRITE_FLAGS)
        if not _path_allowed(path, writing):
            raise ToolRefused(
                ("writing " if writing else "reading ") + repr(str(path))
                + " is not allowed by this tool's capabilities"
            )
        return
    if event in ("socket.connect", "socket.getaddrinfo", "socket.bind",
                 "urllib.Request", "http.client.connect", "ftplib.connect",
                 "smtplib.connect"):
        if not _MAY_NET:
            raise ToolRefused("this tool may not use the network")
        if event == "socket.getaddrinfo" and args:
            host = str(args[0])
            if host and host not in _HOSTS:
                raise ToolRefused("this tool may not reach " + repr(host))
        return
    if event.startswith(("subprocess.", "os.exec", "os.fork", "os.posix_spawn",
                         "os.system", "os.spawn", "pty.spawn", "ctypes.")):
        raise ToolRefused("this tool may not start processes or load native code")
    if event in ("os.kill", "os.killpg"):
        raise ToolRefused("this tool may not signal processes")
    if event in ("os.remove", "os.rename", "os.link", "os.symlink", "os.truncate",
                 "shutil.copyfile", "shutil.move"):
        target = args[0] if args else None
        if target is not None and not _path_allowed(target, True):
            raise ToolRefused(repr(str(target)) + " is not writable by this tool")
        return
    if event in ("os.listdir", "os.scandir", "glob.glob"):
        # Enumerating a directory is not reading a file, so the `open` rule does not
        # see it — but the names alone are worth refusing. `os.stat` is deliberately
        # *not* here: too much of the standard library depends on it.
        target = args[0] if args else None
        if target is not None and not _path_allowed(target, False):
            raise ToolRefused(repr(str(target)) + " is not listable by this tool")
        return


def _install_guard():
    """Apply the limits and arm the hook.

    Called by the harness *after* its own imports and *before* the tool's source runs:
    an audit hook cannot be removed once installed, which is the property that makes it
    worth using — and also means everything the harness needs must already be imported.
    """
    _apply_limits()
    _sys.addaudithook(_hook)
'''


def runtime_paths() -> list[str]:
    """Where the interpreter keeps itself. Readable, because imports are file reads.

    Taken from the running interpreter rather than guessed, because the child runs the
    same one. Nothing here is yours: the project directory is not included, which is
    the part that matters — your ``config.yaml`` is not readable by a tool.
    """
    import sys
    import sysconfig

    candidates = [sys.base_prefix, sys.prefix, "/dev/urandom"]
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        with contextlib.suppress(KeyError):
            candidates.append(sysconfig.get_paths()[key])
    seen: list[str] = []
    for candidate in candidates:
        resolved = str(Path(candidate).resolve())
        if resolved not in seen:
            seen.append(resolved)
    return seen


def guard_source(policy: Policy) -> str:
    """The prelude that enforces ``policy`` from inside the tool's own process."""
    payload = policy.to_dict()
    payload["runtime_paths"] = runtime_paths()
    return GUARD.replace("__POLICY__", repr(json.dumps(payload)))
