"""Chat commands: /forget, /good, /bad, /correct, /clear, /tool, /tools, /help.

These are the user's direct control over what the system learns, so they are handled
before the model ever sees the message and they never need the GPU.

``/tool`` is the odd one out: it does not change what was learned, it changes what the
system *is*. It is here rather than in the model's hands because "write me a tool for
that" is a thing people say in the middle of a sentence, and a request that depends on
the model recognising it would be a request that works four times out of five. Typed as
a command it is recorded, every time, in the same queue the night reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gustav.core.events import Events
from gustav.core.journal import Journal
from gustav.store.db import Database

HELP = """Commands
  /good                 this answer was good — becomes a preference pair
  /bad                  this answer was bad — becomes a preference pair
  /correct <answer>     this is what you should have said — highest-priority training data
  /forget [all]         never train on the last exchange (or this whole session)
  /clear                start a fresh conversation (history is kept in the log)
  /tool <what it does>  have a tool written for it tonight, for you to read and approve
  /tools                what it can call now, what is waiting to be read
  /help                 this list
Anything else is sent to the model."""


@dataclass
class CommandResult:
    reply: str
    action: str
    #: True when the conversation history should be dropped.
    clears_history: bool = False
    data: dict[str, Any] | None = None


@dataclass
class ParsedCommand:
    name: str
    argument: str = ""


def parse(message: str) -> ParsedCommand | None:
    text = message.strip()
    if not text.startswith("/") or len(text) < 2:
        return None
    head, _, rest = text[1:].partition(" ")
    return ParsedCommand(name=head.lower(), argument=rest.strip())


class CommandHandler:
    """Applies a command to the store. Returns None for anything that is not a command."""

    KNOWN = ("good", "bad", "correct", "forget", "clear", "tool", "tools", "help")

    def __init__(
        self,
        db: Database,
        journal: Journal,
        events: Events,
        *,
        request_tool: Callable[[str], dict[str, Any]] | None = None,
        describe_tools: Callable[[], list[str]] | None = None,
    ):
        self.db = db
        self.journal = journal
        self.events = events
        #: Both are the daemon's, and both are optional: a handler built for a test
        #: has no tool store behind it and says so rather than pretending.
        self.request_tool = request_tool
        self.describe_tools = describe_tools

    def handle(self, session_id: int, message: str) -> CommandResult | None:
        command = parse(message)
        if command is None:
            return None
        if command.name not in self.KNOWN:
            return CommandResult(
                reply=f"unknown command /{command.name}\n\n{HELP}", action="unknown"
            )
        return getattr(self, f"_{command.name}")(session_id, command.argument)

    # --- the commands ---------------------------------------------------
    def _help(self, session_id: int, argument: str) -> CommandResult:
        return CommandResult(reply=HELP, action="help")

    def _tool(self, session_id: int, argument: str) -> CommandResult:
        """Ask for a tool. The night writes it; you read it; then it may run."""
        if not argument.strip():
            return CommandResult(
                reply=(
                    "Say what the tool should do, in a sentence:\n"
                    "  /tool raise a number to a large power exactly\n"
                    "  /tool convert a unix timestamp to my timezone\n"
                    "It is written tonight, in a sandbox, and waits for you to read it."
                ),
                action="tool_help",
            )
        if self.request_tool is None:
            return CommandResult(
                reply="this daemon cannot take tool requests (tools are switched off)",
                action="tool_unavailable",
            )
        try:
            result = self.request_tool(argument)
        except Exception as exc:  # noqa: BLE001 - a refusal is an answer, not a crash
            return CommandResult(reply=f"that was not recorded: {exc}", action="tool_failed")
        return CommandResult(
            reply=(
                f"Asked for (request #{result.get('request', '?')}): {argument}\n"
                "It is written in tonight's run, checked in a sandbox, and then waits "
                "in `ai tools` for you to read the source before it can run."
            ),
            action="tool_requested",
            data=result,
        )

    def _tools(self, session_id: int, argument: str) -> CommandResult:
        """What it can call right now — the same list the model is told about."""
        if self.describe_tools is None:
            return CommandResult(reply="tools are not available here", action="tools")
        lines = self.describe_tools()
        return CommandResult(reply="\n".join(lines) or "no tools", action="tools")

    def _clear(self, session_id: int, argument: str) -> CommandResult:
        return CommandResult(
            reply="started a fresh conversation (the log keeps the old one)",
            action="clear",
            clears_history=True,
        )

    def _good(self, session_id: int, argument: str) -> CommandResult:
        return self._feedback(session_id, "good", argument, weight=2)

    def _bad(self, session_id: int, argument: str) -> CommandResult:
        return self._feedback(session_id, "bad", argument, weight=2)

    def _correct(self, session_id: int, argument: str) -> CommandResult:
        if not argument:
            return CommandResult(
                reply="usage: /correct <what I should have said>", action="correct-missing"
            )
        return self._feedback(session_id, "correct", argument, weight=10)

    def _feedback(
        self, session_id: int, kind: str, payload: str, *, weight: int
    ) -> CommandResult:
        turn = self._last_assistant_turn(session_id)
        if turn is None:
            return CommandResult(
                reply="there is no answer of mine to judge yet", action=f"{kind}-no-target"
            )
        feedback_id = self.db.insert(
            "feedback",
            {
                "turn_id": turn["id"],
                "kind": kind,
                "payload": payload or None,
                "source": "user",
                # Explicit user feedback needs no review: the user is the reviewer.
                "reviewed": 1,
            },
        )
        # Corrections are worth more than a thumb, so they count for more pending work.
        self.journal.add_pending_work(f"feedback:{kind}", weight=weight, ref=str(feedback_id))
        self.events.emit(
            f"wake.feedback.{kind}",
            f"{kind} on turn {turn['id']}",
            turn=turn["id"],
            feedback=feedback_id,
        )
        replies = {
            "good": "noted — I will try to do more of that",
            "bad": "noted — I will treat that answer as a negative example",
            "correct": "thank you — that correction is top-priority training data",
        }
        return CommandResult(
            reply=replies[kind], action=kind, data={"turn_id": turn["id"], "feedback_id": feedback_id}
        )

    def _forget(self, session_id: int, argument: str) -> CommandResult:
        whole_session = argument.strip().lower() in {"all", "session", "everything"}
        if whole_session:
            rows = self.db.query(
                "SELECT id FROM turns WHERE session_id = ?", (session_id,)
            )
        else:
            rows = self._last_exchange(session_id)
        if not rows:
            return CommandResult(reply="there is nothing to forget yet", action="forget-empty")

        ids = [row["id"] for row in rows]
        marks = ", ".join("?" for _ in ids)
        self.db.execute(
            f"UPDATE turns SET trainable = 0, exclude_reason = 'user:/forget'"
            f" WHERE id IN ({marks})",
            ids,
        )
        # Pull the queued work for those turns, so sleep never looks at them.
        self.db.execute(
            f"UPDATE pending_work SET consumed_run_id = -1"
            f" WHERE kind = 'turn' AND ref IN ({marks}) AND consumed_run_id IS NULL",
            [str(i) for i in ids],
        )
        self.events.emit(
            "wake.forget",
            f"{len(ids)} turn(s) marked never-train"
            + (" (whole session)" if whole_session else ""),
            turns=ids,
        )
        scope = "this whole conversation" if whole_session else "the last exchange"
        return CommandResult(
            reply=f"forgotten: {scope} will never be trained on ({len(ids)} turns)",
            action="forget",
            clears_history=whole_session,
            data={"turns": ids},
        )

    # --- helpers --------------------------------------------------------
    def _last_assistant_turn(self, session_id: int):
        return self.db.query_one(
            "SELECT * FROM turns WHERE session_id = ? AND role = 'assistant'"
            " ORDER BY id DESC LIMIT 1",
            (session_id,),
        )

    def _last_exchange(self, session_id: int) -> list:
        """The most recent user turn and everything after it."""
        anchor = self.db.query_one(
            "SELECT id FROM turns WHERE session_id = ? AND role = 'user'"
            " ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        if anchor is None:
            return []
        return self.db.query(
            "SELECT id FROM turns WHERE session_id = ? AND id >= ? ORDER BY id",
            (session_id, anchor["id"]),
        )
