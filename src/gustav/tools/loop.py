"""Calling tools during a conversation.

The model asks for a tool the way its own chat template taught it to — Qwen3 and
friends emit ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` — and this
turns that into an actual call, a result, and another turn of generation.

Everything here is about keeping the blast radius small:

* **Only approved tools exist.** The schemas offered to the model come from the store's
  approved set, so a proposed-but-unreviewed tool is not something it can even name. A
  call to anything else comes back as an error the model can read.
* **A budget per answer.** ``tools.max_calls_per_turn`` calls, then the loop stops and
  the model answers with what it has. A model that gets stuck calling the same tool
  cannot turn one question into a night of compute.
* **A failure is told to the model, not hidden.** The error goes back as the tool
  result, so it can apologise, try different arguments, or answer without the tool —
  all of which are better than a retry nobody asked for.
* **Every call is logged** with its arguments and its result, so "why did it say that"
  is answerable afterwards.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from gustav.tools.policy import policy_for
from gustav.tools.runner import ToolResult, ToolRunner
from gustav.tools.store import ToolStore, ToolVersion

#: The marker the chat templates use. Kept as a constant because it is also a stop
#: sequence: generation ends at the closing tag so the tool runs before the model
#: carries on inventing what it returned.
OPEN = "<tool_call>"
CLOSE = "</tool_call>"

_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)
#: Some templates emit the plain JSON without the wrapper when they are unsure.
_BARE = re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def describe(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in self.arguments.items())})"


def parse_calls(text: str) -> list[ToolCall]:
    """Tool calls in a model's output, however tidily it wrapped them."""
    found: list[ToolCall] = []
    seen: set[str] = set()
    candidates = [m.group(1) for m in _CALL.finditer(text or "")]
    if not candidates:
        candidates = [m.group(0) for m in _BARE.finditer(text or "")]
    for raw in candidates:
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                arguments = {}
        if not name or not isinstance(arguments, dict):
            continue
        key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        if key in seen:
            continue
        seen.add(key)
        found.append(ToolCall(name=name, arguments=arguments, raw=raw))
    return found


def strip_calls(text: str) -> str:
    """The answer without the tool-call markup, for what the user is shown."""
    cleaned = _CALL.sub("", text or "")
    if cleaned == text:
        cleaned = _BARE.sub("", cleaned)
    return cleaned.replace(OPEN, "").replace(CLOSE, "").strip()


@dataclass
class ToolUse:
    """What a conversation's tool calls did, for the reply and the log."""

    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [call.name for call in self.calls]

    def describe(self) -> list[str]:
        return [
            f"{call.describe()} -> {result.summary}"
            for call, result in zip(self.calls, self.results, strict=False)
        ]


class ToolSession:
    """The approved tools, for the length of one answer."""

    def __init__(
        self,
        store: ToolStore,
        config,
        *,
        runner: ToolRunner | None = None,
        events: Any = None,
        session_id: int | None = None,
        source: str = "chat",
    ):
        self.store = store
        self.config = config
        self.runner = runner or ToolRunner()
        self.events = events
        self.session_id = session_id
        self.source = source
        self.use = ToolUse()
        self._available: dict[str, ToolVersion] | None = None

    # --- what the model is told ----------------------------------------
    @property
    def available(self) -> dict[str, ToolVersion]:
        if self._available is None:
            self._available = {v.name: v for v in self.store.approved()}
        return self._available

    def schemas(self) -> list[dict[str, Any]]:
        return [version.spec().schema() for version in self.available.values()]

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.available)

    @property
    def remaining(self) -> int:
        return max(self.config.max_calls_per_turn - len(self.use.calls), 0)

    # --- calling --------------------------------------------------------
    def execute(self, call: ToolCall, *, turn_id: int | None = None) -> ToolResult:
        """Run one call. Never raises: a failure is a result the model can read."""
        version = self.available.get(call.name)
        if version is None:
            known = ", ".join(sorted(self.available)) or "none"
            result = ToolResult(
                ok=False,
                kind="unknown",
                error=f"there is no approved tool called {call.name!r}; available: {known}",
            )
        elif self.config.require_approval and version.status != "approved":
            result = ToolResult(
                ok=False, kind="unapproved", error=f"{call.name} has not been approved"
            )
        else:
            policy = policy_for(version.capabilities, self.config)
            result = self.runner.run(version.source, call.arguments, policy)

        self.use.calls.append(call)
        self.use.results.append(result)
        self.store.record_call(
            call.name,
            version=version.version if version else None,
            arguments=call.arguments,
            result=result.result,
            ok=result.ok,
            error=result.summary if not result.ok else "",
            duration_ms=result.duration_ms,
            session_id=self.session_id,
            turn_id=turn_id,
            source=self.source,
        )
        if self.events is not None:
            self.events.emit(
                "tool.called" if result.ok else "tool.failed",
                f"{call.describe()} -> {result.summary}",
                level="info" if result.ok else "warning",
                tool=call.name,
                version=version.version if version else None,
                ok=result.ok,
                ms=result.duration_ms,
            )
        return result

    def message_for(self, call: ToolCall, result: ToolResult) -> dict[str, str]:
        """The tool's answer, in the shape the chat template expects it back."""
        return {"role": "tool", "name": call.name, "content": result.for_model()}
