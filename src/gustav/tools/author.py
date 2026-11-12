"""Asking the model to write a tool.

The model is told the rules it will be judged by — the module allowlist, the
capabilities this installation permits, the ``run`` contract, the requirement to bring
worked examples — and then its proposal is put through exactly those checks. When it
fails, the objections are handed back once so it can repair its own work; a second
failure is reported rather than retried forever.

That repair loop is the only place the model is given a second chance, and it is given
the *machine's* objections, not a human's. The human's attention is spent on a tool
that already compiles, matches its own schema, and does what its examples claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from gustav.learn.jsonish import loads as json_loads
from gustav.tools.policy import CAPABILITIES
from gustav.tools.review import Checks, check
from gustav.tools.runner import ToolRunner
from gustav.tools.spec import ToolSpec

SYSTEM = """\
You write one small Python tool for a local assistant to call. Reply with a single JSON
object and nothing else, with these keys:

  "name"         lower_snake_case, a verb phrase, e.g. "convert_units"
  "description"  one sentence saying when the assistant should call it
  "parameters"   a JSON Schema object: {"type":"object","properties":{...},"required":[...]}
                 every property needs a "type" and a "description"
  "capabilities" a list from: %(capabilities)s
  "source"       Python defining `run(...)` with exactly the parameters in the schema
  "examples"     1 to 4 objects: {"arguments": {...}, "expect": <the exact return value>}

Hard rules, because your tool is checked against them before anyone reads it:

- `run` must take named parameters only. No *args, no **kwargs.
- `run` must return something JSON can represent: numbers, strings, booleans, lists,
  dicts, null. Not objects, not dates — format those as strings.
- You may import only: %(modules)s
- No network, no files, no subprocesses, no clever tricks to get them. The sandbox
  refuses them at runtime and the review refuses them before that.
- Your examples must be *exactly* what `run` returns, and they must fail if `run` were
  emptied out. An example like {"arguments":{...},"expect":null} proves nothing.
- Prefer a tool that does one thing with no state. Raise ValueError with a clear
  message for bad input; do not return error strings.
"""

REPAIR = """\
Your tool was refused by the checks. Fix exactly these problems and reply with the
whole JSON object again:

%(problems)s
"""


@dataclass
class AuthoredTool:
    spec: ToolSpec | None = None
    checks: Checks | None = None
    attempts: int = 0
    reason: str = ""
    raw: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.spec is not None and self.checks is not None and self.checks.ok

    def describe(self) -> str:
        if self.ok and self.spec is not None:
            return f"wrote {self.spec.name} in {self.attempts} attempt(s)"
        return f"no tool: {self.reason}"


def system_prompt(config) -> str:
    return SYSTEM % {
        "capabilities": ", ".join(
            f"{name} ({CAPABILITIES[name]})" for name in config.allow_capabilities
            if name in CAPABILITIES
        ),
        "modules": ", ".join(sorted(config.allowed_modules)),
    }


def _spec_from(payload: dict[str, Any]) -> ToolSpec | None:
    name = str(payload.get("name") or "").strip()
    source = str(payload.get("source") or "").strip()
    if not name or not source:
        return None
    examples = payload.get("examples") or []
    if isinstance(examples, dict):
        examples = [examples]
    capabilities = payload.get("capabilities") or ["pure"]
    if isinstance(capabilities, str):
        capabilities = [capabilities]
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    return ToolSpec(
        name=name,
        description=str(payload.get("description") or "").strip(),
        source=source,
        parameters=parameters,
        capabilities=tuple(str(c) for c in capabilities),
        examples=tuple(e for e in examples if isinstance(e, dict)),
    )


def write_tool(
    model: Any,
    request: str,
    config,
    *,
    runner: ToolRunner | None = None,
    attempts: int = 2,
    max_tokens: int = 1200,
) -> AuthoredTool:
    """Ask for a tool, check it, and let it repair itself once."""
    outcome = AuthoredTool()
    if not getattr(model, "available", False):
        outcome.reason = getattr(model, "error", None) or "no model available"
        return outcome

    runner = runner or ToolRunner()
    system = system_prompt(config)
    prompt = f"Write a tool for this: {request.strip()}"

    for attempt in range(1, max(attempts, 1) + 1):
        outcome.attempts = attempt
        answer = model.ask(prompt, system=system, max_tokens=max_tokens)
        outcome.raw.append(answer or "")
        payload = json_loads(answer or "")
        if not isinstance(payload, dict):
            outcome.reason = "the model did not reply with a JSON object"
            prompt = (
                f"Write a tool for this: {request.strip()}\n\n"
                + REPAIR % {"problems": "- the reply was not a single JSON object"}
            )
            continue
        spec = _spec_from(payload)
        if spec is None:
            outcome.reason = "the reply had no name or no source"
            prompt = (
                f"Write a tool for this: {request.strip()}\n\n"
                + REPAIR % {"problems": "- the reply needs both a name and a source"}
            )
            continue

        checks = check(spec, config, runner=runner)
        outcome.spec, outcome.checks = spec, checks
        if checks.ok:
            outcome.reason = ""
            return outcome
        outcome.reason = "; ".join(checks.problems)
        problems = "\n".join(f"- {problem}" for problem in checks.problems)
        for example in checks.examples:
            if not example.ok:
                problems += f"\n- example {example.describe()}"
        prompt = (
            f"Write a tool for this: {request.strip()}\n\n"
            f"Your previous attempt:\n{json.dumps(payload)[:2000]}\n\n"
            + REPAIR % {"problems": problems}
        )
    return outcome
