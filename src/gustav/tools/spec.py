"""What a tool is, as data.

A tool is a name, a description, a JSON schema for its arguments, a capability list,
some examples, and a Python source file that defines ``run``. Everything here is
derivable from or checkable against that source *without executing it*, which is the
point: the first pass over a tool the model just wrote should not involve running it.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from gustav.learn.fingerprint import fingerprint

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
#: JSON Schema types a tool parameter may declare. Anything else is refused: the schema
#: is shown to the model, and a type it cannot produce is a trap rather than a feature.
TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object"})


@dataclass
class Parameter:
    name: str
    required: bool
    default: Any = None


@dataclass
class ToolSpec:
    name: str
    description: str
    source: str
    parameters: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ("pure",)
    examples: tuple[dict[str, Any], ...] = ()
    author: str = "model"

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.name, self.source, json.dumps(self.parameters, sort_keys=True))

    def schema(self) -> dict[str, Any]:
        """The tool definition in the shape a chat template's ``tools=`` expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def signature(self) -> list[Parameter] | None:
        """``run``'s parameters, read from the source. None if there is no ``run``."""
        return signature_of(self.source)

    def describe(self) -> str:
        args = ", ".join(sorted((self.parameters.get("properties") or {})))
        return f"{self.name}({args}) — {self.description}"


def signature_of(source: str) -> list[Parameter] | None:
    """Parse ``def run(...)`` out of a tool's source. No execution."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            args = node.args
            if args.vararg or args.kwarg:
                # `**kwargs` would make the schema unverifiable: anything would "match".
                return None
            positional = [*args.posonlyargs, *args.args]
            defaults = list(args.defaults)
            required = len(positional) - len(defaults)
            found = [
                Parameter(name=arg.arg, required=index < required)
                for index, arg in enumerate(positional)
            ]
            for index, arg in enumerate(args.kwonlyargs):
                default = args.kw_defaults[index]
                found.append(Parameter(name=arg.arg, required=default is None))
            return found
    return None


def imports_of(source: str) -> set[str]:
    """Every module the source imports, at any nesting depth."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def stub_of(source: str) -> str:
    """The same module with ``run`` emptied out.

    Used to prove the examples test something: with ``run`` returning None, they have
    to fail. A tool whose examples pass without an implementation is a tool that
    measures nothing — the same check the practice exercises get.
    """
    parameters = signature_of(source)
    if parameters is None:
        return "def run(*args, **kwargs):\n    return None\n"
    rendered = ", ".join(
        p.name if p.required else f"{p.name}=None" for p in parameters
    )
    return f"def run({rendered}):\n    return None\n"


def validate_schema(parameters: dict[str, Any], source: str) -> list[str]:
    """Problems with the declared schema, as sentences. Empty means it is sound."""
    problems: list[str] = []
    if not isinstance(parameters, dict):
        return ["the parameters must be a JSON Schema object"]
    if parameters.get("type") not in (None, "object"):
        problems.append("the parameters schema must be of type 'object'")
    properties = parameters.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, dict):
        return ["the schema's 'properties' must be an object"]
    required = parameters.get("required") or []
    if not isinstance(required, list):
        problems.append("the schema's 'required' must be a list")
        required = []

    for key, value in properties.items():
        if not isinstance(value, dict):
            problems.append(f"the schema for {key!r} is not an object")
            continue
        declared = value.get("type")
        if declared is not None and declared not in TYPES:
            problems.append(f"{key!r} declares an unsupported type {declared!r}")
        if not str(value.get("description") or "").strip():
            problems.append(f"{key!r} has no description, so the model cannot use it well")

    signature = signature_of(source)
    if signature is None:
        problems.append(
            "the source defines no run() with a fixed signature (*args/**kwargs are "
            "not allowed, because then the schema could not be checked)"
        )
        return problems

    names = {p.name for p in signature}
    for key in properties:
        if key not in names:
            problems.append(f"the schema declares {key!r}, which run() does not accept")
    for parameter in signature:
        if parameter.name not in properties:
            problems.append(
                f"run() takes {parameter.name!r}, which the schema does not declare"
            )
        if parameter.required and parameter.name not in required:
            problems.append(
                f"{parameter.name!r} has no default, so the schema must list it as required"
            )
        if not parameter.required and parameter.name in required:
            problems.append(
                f"{parameter.name!r} has a default, so the schema should not require it"
            )
    for key in required:
        if key not in properties:
            problems.append(f"{key!r} is required but not declared")
    return problems


def validate_name(name: str) -> str | None:
    if not name or not NAME_RE.match(name):
        return (
            f"{name!r} is not a usable tool name: lower case, starting with a letter, "
            f"letters, digits and underscores, 2 to 41 characters"
        )
    return None
