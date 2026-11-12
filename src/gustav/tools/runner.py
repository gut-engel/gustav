"""Running one tool call.

The protocol is deliberately dull: the source and the arguments go in as JSON on stdin,
one JSON object comes back on stdout, and anything else the process printed is captured
separately. The source travels *in the request* rather than in the script, so the file
the interpreter runs is the fixed harness and the tool's own code is executed after the
guard is armed.
A tool cannot return a Python object, hold state between calls, or talk to the daemon —
it is a function from JSON to JSON that happens to run in its own process.

Everything about a call is recorded: the arguments, the result, how long it took, and
what the sandbox refused. A tool that starts failing is visible in `ai tools` rather
than quietly degrading answers.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gustav.tools.policy import Policy, guard_source

#: Wraps the tool's own source: the guard first, then the tool, then the call.
HARNESS = '''\
import json as _json, sys as _sys, traceback as _traceback

__GUARD__

def _fail(kind, message, detail=None):
    _sys.stdout.write(_json.dumps(
        {"ok": False, "error": message, "kind": kind, "detail": detail}
    ))
    _sys.stdout.flush()
    raise SystemExit(0)

try:
    _request = _json.loads(_sys.stdin.read() or "{}")
except Exception as _exc:
    _fail("protocol", "the arguments were not valid JSON: %s" % _exc)

# Everything the harness itself needs is imported above: from here on the guard is
# armed and the tool's own code is subject to it.
_install_guard()

try:
    exec(compile(_request["source"], "<tool>", "exec"), globals())
except ToolRefused as _exc:
    _fail("refused", str(_exc))
except SyntaxError as _exc:
    _fail("syntax", "the tool does not compile: %s" % _exc)
except Exception as _exc:
    _fail("import", "%s: %s" % (type(_exc).__name__, _exc), _traceback.format_exc(limit=4))

try:
    run
except NameError:
    _fail("contract", "this tool defines no run() function")

try:
    _result = run(**(_request.get("arguments") or {}))
except TypeError as _exc:
    _fail("arguments", "run() rejected these arguments: %s" % _exc)
except ToolRefused as _exc:
    _fail("refused", str(_exc))
except SystemExit:
    raise
except BaseException as _exc:
    _fail("raised", "%s: %s" % (type(_exc).__name__, _exc),
          _traceback.format_exc(limit=4))

try:
    _payload = _json.dumps({"ok": True, "result": _result})
except (TypeError, ValueError) as _exc:
    _fail("result", "run() returned something that is not JSON: %s" % _exc)

_sys.stdout.write(_payload)
_sys.stdout.flush()
'''


@dataclass
class ToolResult:
    ok: bool
    result: Any = None
    error: str = ""
    kind: str = ""
    detail: str = ""
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False

    @property
    def summary(self) -> str:
        if self.ok:
            return "ok"
        if self.timed_out:
            return f"timed out after {self.duration_ms} ms"
        return f"{self.kind or 'error'}: {self.error}"[:400]

    def for_model(self) -> str:
        """What the model is shown. A failure is information, not a hidden retry."""
        if self.ok:
            try:
                return json.dumps(self.result, ensure_ascii=False)[:4000]
            except (TypeError, ValueError):
                return str(self.result)[:4000]
        return json.dumps({"error": self.summary})


class ToolRunner:
    def __init__(self, *, python: str | None = None, workspace_root: Path | None = None):
        self.python = python or sys.executable
        self.workspace_root = workspace_root

    def program(self, policy: Policy) -> str:
        """The harness, with this policy's guard compiled into it."""
        return HARNESS.replace("__GUARD__", guard_source(policy))

    def run(
        self, source: str, arguments: dict[str, Any], policy: Policy
    ) -> ToolResult:
        """Call ``run(**arguments)`` from ``source`` under ``policy``."""
        root = self.workspace_root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="gustav-tool-", dir=str(root) if root else None
        ) as tmp:
            workspace = Path(tmp)
            # The guard needs the resolved workspace path, so the policy is finished
            # here rather than by the caller.
            scoped = Policy(
                capabilities=policy.capabilities,
                allowed_modules=policy.allowed_modules,
                readable_paths=policy.readable_paths,
                allowed_hosts=policy.allowed_hosts,
                workspace=workspace.resolve(),
                timeout_seconds=policy.timeout_seconds,
                memory_limit_mb=policy.memory_limit_mb,
                max_output_bytes=policy.max_output_bytes,
            )
            script = workspace / "harness.py"
            script.write_text(self.program(scoped), encoding="utf-8")
            return self._spawn(script, workspace, source, arguments, scoped)

    def _spawn(
        self,
        script: Path,
        workspace: Path,
        source: str,
        arguments: dict[str, Any],
        policy: Policy,
    ) -> ToolResult:
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(workspace),
            "TMPDIR": str(workspace),
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                [self.python, "-I", str(script)],
                cwd=str(workspace),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            return ToolResult(
                ok=False, kind="sandbox", error=f"could not start the tool: {exc}"
            )

        timed_out = False
        try:
            stdout, stderr = process.communicate(
                json.dumps({"source": source, "arguments": arguments}),
                timeout=policy.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(process)
            stdout, stderr = process.communicate()
        duration_ms = int((time.monotonic() - started) * 1000)
        limit = policy.max_output_bytes
        stdout, stderr = (stdout or "")[:limit], (stderr or "")[:limit]

        if timed_out:
            return ToolResult(
                ok=False, kind="timeout", timed_out=True, duration_ms=duration_ms,
                error=f"the tool ran longer than {policy.timeout_seconds:.0f}s",
                stdout=stdout, stderr=stderr,
            )

        payload = _last_json(stdout)
        if payload is None:
            return ToolResult(
                ok=False,
                kind="protocol",
                error=(
                    "the tool printed no result"
                    + (f"; it exited with {process.returncode}" if process.returncode else "")
                ),
                detail=stderr[-800:],
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
            )
        if payload.get("ok"):
            return ToolResult(
                ok=True, result=payload.get("result"), stdout=stdout, stderr=stderr,
                duration_ms=duration_ms,
            )
        return ToolResult(
            ok=False,
            kind=str(payload.get("kind") or "error"),
            error=str(payload.get("error") or "the tool failed"),
            detail=str(payload.get("detail") or "")[:2000],
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except ProcessLookupError:
                pass


def _last_json(text: str) -> dict[str, Any] | None:
    """The result object, even if the tool printed other things first.

    A tool that prints progress and then its result is well-behaved enough; a tool that
    prints nothing parseable has failed, and that is what the caller is told.
    """
    text = (text or "").strip()
    if not text:
        return None
    for candidate in (text, text.splitlines()[-1] if text.splitlines() else ""):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed
    start = text.rfind('{"ok"')
    if start >= 0:
        try:
            parsed = json.loads(text[start:])
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
