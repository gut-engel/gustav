"""The checks a proposed tool has to pass before anyone is asked to read it.

The user's attention is the scarce resource here. A review queue full of tools that do
not compile, do not match their own schema, or "pass" examples that assert nothing is a
queue nobody reads — so the cheap, mechanical objections are raised by the machine, and
what reaches you is a tool that at least does what it says on its own examples.

Six checks, in the order that costs least first:

1. the name and description are usable;
2. the source compiles, defines ``run`` with a fixed signature, and is not enormous;
3. every import is on the allowlist (the runtime guard enforces this too — this is so
   the refusal arrives with a readable reason instead of at the first call);
4. the declared capabilities are ones you allow;
5. the declared schema matches ``run``'s signature, parameter by parameter;
6. the examples run in the sandbox and produce what they claim — **and** the same
   examples fail when ``run`` is stubbed out, which is what makes step 6 mean anything.

Passing all six is not approval. It is the difference between "worth reading" and
"noise".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from gustav.tools.policy import NEVER_IMPORTABLE, policy_for, unsupported
from gustav.tools.runner import ToolRunner
from gustav.tools.spec import (
    ToolSpec,
    imports_of,
    signature_of,
    stub_of,
    validate_name,
    validate_schema,
)

#: A tool has to come with at least this many worked examples.
MIN_EXAMPLES = 1
MAX_EXAMPLES = 8


@dataclass
class ExampleResult:
    arguments: dict[str, Any]
    expected: Any
    got: Any = None
    ok: bool = False
    detail: str = ""

    def describe(self) -> str:
        mark = "ok" if self.ok else "failed"
        return f"{mark}: run({_args(self.arguments)}) -> {self.got!r}" + (
            f" (expected {self.expected!r})" if not self.ok else ""
        ) + (f" [{self.detail}]" if self.detail else "")


@dataclass
class Checks:
    problems: list[str] = field(default_factory=list)
    examples: list[ExampleResult] = field(default_factory=list)
    stub_passes: bool = False
    imports: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "imports": list(self.imports),
            "capabilities": list(self.capabilities),
            "stub_passes": self.stub_passes,
            "examples": [
                {
                    "arguments": e.arguments,
                    "expected": e.expected,
                    "got": e.got,
                    "ok": e.ok,
                    "detail": e.detail,
                }
                for e in self.examples
            ],
        }

    def describe(self) -> list[str]:
        lines = [f"{'passed' if self.ok else 'REFUSED'} the admission checks"]
        lines += [f"  - {problem}" for problem in self.problems]
        if self.imports:
            lines.append("  imports: " + ", ".join(sorted(self.imports)))
        for example in self.examples:
            lines.append(f"  example {example.describe()}")
        return lines


def _args(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in sorted(arguments.items()))


def _comparable(value: Any) -> Any:
    """Compare the way JSON would, so 2 and 2.0 agree and dict order does not matter."""
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError):
        return value


def check(
    spec: ToolSpec,
    config,
    *,
    runner: ToolRunner | None = None,
    run_examples: bool = True,
) -> Checks:
    """Everything that can be objected to without a human reading the source."""
    checks = Checks(capabilities=list(spec.capabilities))
    runner = runner or ToolRunner()

    if (problem := validate_name(spec.name)) is not None:
        checks.problems.append(problem)
    if len(spec.description.strip()) < 8:
        checks.problems.append(
            "the description is too short to tell the model when to use this tool"
        )
    if len(spec.source.encode("utf-8")) > config.max_source_bytes:
        checks.problems.append(
            f"the source is larger than tools.max_source_bytes "
            f"({config.max_source_bytes} bytes)"
        )
    if signature_of(spec.source) is None:
        checks.problems.append(
            "the source must define run() with named parameters and no *args/**kwargs"
        )

    checks.imports = sorted(imports_of(spec.source))
    allowed = set(config.allowed_modules)
    for module in checks.imports:
        root = module.split(".")[0]
        if root in NEVER_IMPORTABLE and module not in allowed:
            checks.problems.append(
                f"it imports {module!r}, which a tool may never use"
            )
        elif not any(
            module == name or module.startswith(name + ".") for name in allowed
        ):
            checks.problems.append(
                f"it imports {module!r}, which is not in tools.allowed_modules"
            )

    if refused := unsupported(spec.capabilities, config.allow_capabilities):
        checks.problems.append(
            "it asks for capabilities this installation does not allow: "
            + ", ".join(refused)
        )

    checks.problems.extend(validate_schema(spec.parameters, spec.source))

    if not spec.examples:
        checks.problems.append(
            f"it comes with no examples; at least {MIN_EXAMPLES} is required so the "
            f"tool can be checked against its own claims"
        )
    elif len(spec.examples) > MAX_EXAMPLES:
        checks.problems.append(f"more than {MAX_EXAMPLES} examples is too many to review")

    for index, example in enumerate(spec.examples):
        if not isinstance(example, dict) or "arguments" not in example:
            checks.problems.append(
                f"example {index + 1} needs an 'arguments' object and an 'expect' value"
            )

    # Everything above is free. Running the examples is not, and running them against a
    # tool that already failed a check would just produce a second, less useful reason.
    if not run_examples or checks.problems:
        return checks

    policy = policy_for(spec.capabilities, config)
    for example in spec.examples:
        arguments = dict(example.get("arguments") or {})
        expected = example.get("expect")
        outcome = runner.run(spec.source, arguments, policy)
        result = ExampleResult(arguments=arguments, expected=expected)
        if not outcome.ok:
            result.detail = outcome.summary
            result.got = None
        else:
            result.got = outcome.result
            result.ok = _comparable(outcome.result) == _comparable(expected)
        checks.examples.append(result)

    failed = [e for e in checks.examples if not e.ok]
    if failed:
        checks.problems.append(
            f"{len(failed)} of {len(checks.examples)} examples did not produce what "
            f"the tool claims they would"
        )
        return checks

    # The check that makes the others worth having: with run() emptied out, the
    # examples must fail. If they still pass, they assert nothing.
    stub = stub_of(spec.source)
    stub_results = [
        runner.run(stub, dict(example.get("arguments") or {}), policy)
        for example in spec.examples
    ]
    checks.stub_passes = all(
        outcome.ok and _comparable(outcome.result) == _comparable(example.get("expect"))
        for outcome, example in zip(stub_results, spec.examples, strict=True)
    )
    if checks.stub_passes:
        checks.problems.append(
            "its examples pass even with run() emptied out, so they do not test the "
            "tool at all"
        )
    return checks
