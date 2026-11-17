"""What gustav tells the model about itself.

A model that has never been told what it is attached to will answer as the generic
assistant it was trained to be. Asked for a tool, it says it does not have one and
cannot make one — which is wrong here, and wrong in a way that costs the user the
feature: tools are written overnight, sandboxed, and shown to them to approve. Asked
for maths it cannot do, it writes LaTeX, which the chat page renders as raw ``$`` signs.

Both are the same bug. The system prompt was two sentences and said nothing true about
this machine. So this module builds one out of what is actually the case right now:

* which tools are approved and callable **this turn**,
* which have been written and are waiting for someone to read them,
* whether the night may write new ones at all,
* what the person can type, and
* what the answer is rendered by, so the model stops writing maths nobody can read.

It reads the tool store and the config and nothing else, and every number in it comes
from the same tables the ``/tools`` panel shows. If a capability is switched off in
config.yaml, the prompt stops claiming it — a prompt that promises what the daemon
refuses to do is worse than one that promises nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Kept as the fallback for anything that has no daemon behind it — the training
#: dataset builder uses it, and the tests of the context budget do too.
DEFAULT_SYSTEM_PROMPT = (
    "You are Gustav, a local assistant that learns from the conversations it has. "
    "Answer precisely and admit what you do not know."
)

#: What the person can type into the box. Kept here rather than imported from
#: `wake.commands` so the prompt is one paragraph, not a command listing.
COMMANDS = "/good, /bad, /correct <answer>, /forget, /tool <what it should do>, /help"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""

    def line(self) -> str:
        return f"{self.name} — {self.description}" if self.description else self.name


@dataclass(frozen=True)
class Capabilities:
    """Everything the prompt is allowed to claim, as facts rather than adjectives."""

    instance: str = "gustav"
    base_model: str = ""
    #: Tools approved, active, and reachable from this backend on this turn.
    callable_tools: tuple[Tool, ...] = ()
    #: Written and waiting for a human to read the source.
    pending_tools: tuple[Tool, ...] = ()
    #: Asked for and not written yet.
    open_requests: tuple[str, ...] = ()
    tools_enabled: bool = True
    #: Whether this backend can call a tool at all (the echo responder cannot).
    can_call_tools: bool = True
    #: Whether the night is allowed to write new ones.
    authoring: bool = True
    require_approval: bool = True
    learning_paused: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    # --- the sentences ---------------------------------------------------
    def tools_paragraph(self) -> str:
        if not self.tools_enabled:
            return (
                "Tools are switched off on this machine, so you can only answer from "
                "what you know. Say so plainly rather than guessing at a number."
            )
        if self.callable_tools and self.can_call_tools:
            listed = "\n".join(f"  - {tool.line()}" for tool in self.callable_tools)
            return f"Tools you can call in this answer:\n{listed}"
        if self.callable_tools and not self.can_call_tools:
            listed = ", ".join(tool.name for tool in self.callable_tools)
            return (
                f"Tools exist here ({listed}) but this model cannot call them directly. "
                f"Tell the person they can run one with `ai tools run <name>`."
            )
        return "No tools have been approved yet, so there are none for you to call."

    def growth_paragraph(self) -> str:
        """The paragraph the whole module exists for."""
        if not self.authoring:
            return (
                "New tools are not being written on this machine (tools.allow_authoring "
                "is off). If something is beyond you, say so plainly and say that the "
                "owner can turn tool authoring back on in config.yaml."
            )
        review = (
            " It is written in a sandbox, checked, and shown to them to read before it "
            "can run."
            if self.require_approval
            else " It is written in a sandbox and checked before it can run."
        )
        waiting = ""
        if self.pending_tools:
            listed = "; ".join(tool.line() for tool in self.pending_tools)
            waiting = (
                f" Already written and waiting for them to read it: {listed}. Say so if "
                f"it is what they are asking for."
            )
        queued = ""
        if self.open_requests:
            listed = "; ".join(self.open_requests)
            queued = f" Already asked for and not written yet: {listed}."
        return (
            "You can be given new abilities, so never say that you cannot be extended. "
            "When something is beyond you — exact or very large arithmetic, dates and "
            "timezones, unit conversion, parsing, anything fiddly and repeatable — say "
            "plainly that you cannot do it reliably in your head, then offer the thing "
            "that actually works: a tool, written for you tonight. The person asks for "
            "one by typing `/tool <what it should do>`; you can say exactly what to "
            f"type.{review}{waiting}{queued}"
        )

    def learning_paragraph(self) -> str:
        if self.learning_paused:
            return (
                "Learning is paused right now, so this conversation is stored but will "
                "not be trained on until it is resumed."
            )
        return (
            "This conversation is what you study tonight: facts are extracted from it, "
            "weak topics are practised, and the adapters you answer with are retrained "
            "on what was said. A confident guess today is tomorrow's training data, so "
            "say what you actually know and mark the rest as uncertain."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "callable_tools": [tool.name for tool in self.callable_tools],
            "pending_tools": [tool.name for tool in self.pending_tools],
            "open_requests": list(self.open_requests),
            "tools_enabled": self.tools_enabled,
            "can_call_tools": self.can_call_tools,
            "authoring": self.authoring,
            "learning_paused": self.learning_paused,
        }


#: How the answer is displayed, which the model has no way of knowing and keeps getting
#: wrong. The LaTeX sentence is here because a model asked for 457^8000000000 writes
#: `$457^{8000000000}$`, and the chat page shows exactly those characters.
RENDERING = (
    "Your answers are rendered as Markdown: use fenced code blocks (```) for code, "
    "commands and output, `backticks` for names, ** for emphasis, and real lists. Do "
    "not use LaTeX or $...$ maths — it is shown as raw characters. Write maths as code "
    "instead: 457 ** 8000000000."
)

STYLE = (
    "Answer the question that was asked, and start with the answer. Never open with "
    "what you are — no \"As a model…\", no \"As an AI…\", no listing of your own "
    "abilities unless someone asks for it. Be direct and short. Admit what you do not "
    "know instead of filling the gap, and never invent a number you could get from a "
    "tool."
)


def brief_prompt(capabilities: Capabilities) -> str:
    """The same claims in a quarter of the words, for a small context window.

    The full prompt is around 420 tokens. On the 4 GB profile the whole window is
    2048, and a fifth of it spent describing the machine is a fifth that cannot hold
    the conversation. Nothing is dropped that changes what the model will *do* — the
    tools it can call, that it can be given more, and what renders the answer — only
    the explanation of why.
    """
    lines = [
        f"You are Gustav, a local assistant running as "
        f"'{capabilities.instance or 'gustav'}' on this machine. No internet of your "
        f"own. This conversation is studied tonight"
        + (" (learning is paused)." if capabilities.learning_paused else "."),
    ]
    if capabilities.tools_enabled and capabilities.callable_tools:
        listed = "; ".join(tool.line() for tool in capabilities.callable_tools)
        lines.append(f"Tools you can call: {listed}.")
    elif capabilities.tools_enabled:
        lines.append("No tools are approved yet.")
    if capabilities.authoring:
        waiting = (
            f" Waiting to be approved: "
            f"{', '.join(tool.name for tool in capabilities.pending_tools)}."
            if capabilities.pending_tools
            else ""
        )
        lines.append(
            "You can be given new abilities: when something is beyond you (exact or "
            "large arithmetic, dates, conversions, parsing), say so and tell them to "
            "type `/tool <what it should do>` — it is written tonight and shown to "
            f"them before it runs. Never say you cannot be extended.{waiting}"
        )
    lines.append(
        "Answers render as Markdown: fenced code blocks, no LaTeX or $...$ (it shows "
        "as raw characters). Write maths as code: 457 ** 8000000000."
    )
    lines.append(
        "Start with the answer. Never open with what you are (\"As a model…\") or "
        "list your own abilities unless asked. Admit what you do not know."
    )
    return "\n\n".join(lines)


def system_prompt(capabilities: Capabilities, *, live: bool = True) -> str:
    """The system message the model is given, built from what is true right now.

    ``live=False`` leaves out the half that changes between one night and the next —
    which tools exist this minute — and is what the training set is built with. The
    model is then trained against the same words it is asked against, minus a list
    that would be wrong by the time the adapter is used.
    """
    name = capabilities.instance or "gustav"
    opening = (
        f"You are Gustav, a local assistant running as '{name}' on this machine, from "
        f"local weights. You have no internet access of your own: anything from outside "
        f"this machine reaches you through a tool or through what the person tells you."
    )
    parts = [
        opening,
        capabilities.learning_paragraph(),
        capabilities.tools_paragraph() if live else "",
        capabilities.growth_paragraph(),
        f"The person can type: {COMMANDS}. These are handled before you see them.",
        RENDERING,
        STYLE,
    ]
    parts.extend(capabilities.notes)
    return "\n\n".join(part for part in parts if part)


def collect(
    config: Any,
    *,
    store: Any = None,
    instance: str = "",
    base_model: str = "",
    can_call_tools: bool = True,
    learning_paused: bool = False,
    max_listed: int = 12,
) -> Capabilities:
    """Read the tool store, and fall back to "no tools" rather than to a guess.

    A broken or missing store must not take the chat down with it: the prompt simply
    stops mentioning tools, which is the same thing it says on a machine that has none.
    """
    tools_config = getattr(config, "tools", None)
    enabled = bool(getattr(tools_config, "enabled", False))
    authoring = bool(getattr(tools_config, "allow_authoring", False))
    require_approval = bool(getattr(tools_config, "require_approval", True))

    callable_tools: tuple[Tool, ...] = ()
    pending_tools: tuple[Tool, ...] = ()
    requests: tuple[str, ...] = ()
    if store is not None and enabled:
        try:
            callable_tools = tuple(
                Tool(version.name, version.description)
                for version in store.approved()[:max_listed]
            )
            pending_tools = tuple(
                Tool(version.name, version.description)
                for version in store.pending()[:max_listed]
            )
            requests = tuple(
                str(row.get("text", ""))[:120]
                for row in store.open_requests(limit=max_listed)
            )
        except Exception:  # noqa: BLE001 - a prompt is not worth failing a chat over
            callable_tools, pending_tools, requests = (), (), ()

    return Capabilities(
        instance=instance or getattr(getattr(config, "instance", None), "name", "") or "gustav",
        base_model=base_model,
        callable_tools=callable_tools,
        pending_tools=pending_tools,
        open_requests=requests,
        tools_enabled=enabled,
        can_call_tools=can_call_tools,
        authoring=authoring,
        require_approval=require_approval,
        learning_paused=learning_paused,
    )


def summary_lines(capabilities: Capabilities) -> list[str]:
    """What `/tools` answers with, and what `ai status` could print.

    The same facts as the prompt, written for a person rather than for a model.
    """
    lines: list[str] = []
    if not capabilities.tools_enabled:
        return ["Tools are switched off on this machine (tools.enabled in config.yaml)."]
    if capabilities.callable_tools:
        lines.append("I can call these right now:")
        lines += [f"  {tool.line()}" for tool in capabilities.callable_tools]
    else:
        lines.append("No tools are approved yet, so I cannot call any.")
    if capabilities.pending_tools:
        lines.append("Written and waiting for you to read them (`ai tools`):")
        lines += [f"  {tool.line()}" for tool in capabilities.pending_tools]
    if capabilities.open_requests:
        lines.append("Asked for, not written yet:")
        lines += [f"  {text}" for text in capabilities.open_requests]
    if capabilities.authoring:
        lines.append("Ask for another with:  /tool <what it should do>")
    else:
        lines.append("New tools are not being written (tools.allow_authoring is off).")
    return lines
