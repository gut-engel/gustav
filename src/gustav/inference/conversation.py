"""Conversation state and the context budget.

The model has a finite window and the user does not care. This module decides what to
keep: the system prompt always, the newest turns always, and as much history in between
as fits — leaving room for the answer, which is the part people forget.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, str]
TokenCounter = Callable[[str], int]

DEFAULT_SYSTEM_PROMPT = (
    "You are Gustav, a local assistant that learns from the conversations it has. "
    "Answer precisely and admit what you do not know."
)


@dataclass
class Turn:
    role: str
    content: str
    turn_id: int | None = None

    def as_message(self) -> Message:
        return {"role": self.role, "content": self.content}


@dataclass
class FitResult:
    messages: list[Message]
    prompt_tokens: int
    dropped_turns: int = 0
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ContextBudget:
    """How many tokens the prompt may use."""

    max_seq_len: int
    #: Room kept for the answer itself.
    reserve_for_completion: int = 512
    #: Safety margin for template tokens the counter cannot see.
    overhead: int = 64

    @property
    def prompt_limit(self) -> int:
        return max(self.max_seq_len - self.reserve_for_completion - self.overhead, 64)

    def fit(
        self,
        messages: Sequence[Message],
        count_tokens: TokenCounter,
        *,
        summarize: Callable[[Sequence[Message]], str] | None = None,
    ) -> FitResult:
        """Drop the oldest turns until the prompt fits; never the system or the last turn.

        ``summarize`` is the hook stage 5 fills in: instead of discarding old turns it
        can fold them into one summary message. Without it, old turns are dropped.
        """
        limit = self.prompt_limit
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        costs = [count_tokens(m["content"]) + 4 for m in rest]
        system_cost = sum(count_tokens(m["content"]) + 4 for m in system)

        def fill(budget: int) -> tuple[list[Message], int]:
            """Keep as many of the newest turns as fit in ``budget``."""
            kept: list[Message] = []
            total = system_cost
            for message, cost in zip(reversed(rest), reversed(costs), strict=True):
                if total + cost > budget and kept:
                    break
                kept.append(message)
                total += cost
            kept.reverse()
            return kept, total

        kept, total = fill(limit)
        dropped_count = len(rest) - len(kept)

        notes: list[str] = []
        truncated = False
        if dropped_count and summarize is not None:
            # Summarising costs tokens, so reserve them *before* filling, or the summary
            # can never fit and the hook would silently do nothing.
            summary = summarize(rest[:dropped_count])
            if summary:
                message = {"role": "system", "content": f"Earlier context: {summary}"}
                cost = count_tokens(message["content"]) + 4
                refilled, refilled_total = fill(max(limit - cost, system_cost))
                kept = [message, *refilled]
                total = refilled_total + cost
                dropped_count = len(rest) - len(refilled)
                notes.append(f"summarised {dropped_count} older turn(s)")
        elif dropped_count:
            notes.append(f"dropped {dropped_count} older turn(s) to fit the context window")

        # Still too long: the newest message alone exceeds the window.
        if total > limit and kept:
            last = kept[-1]
            budget = max(limit - (total - costs[-1] if costs else 0), 256)
            trimmed = _truncate_middle(last["content"], budget, count_tokens)
            if trimmed != last["content"]:
                kept[-1] = {**last, "content": trimmed}
                truncated = True
                notes.append("truncated the middle of the newest message")
                total = system_cost + sum(count_tokens(m["content"]) + 4 for m in kept)

        return FitResult(
            messages=system + kept,
            prompt_tokens=total,
            dropped_turns=dropped_count,
            truncated=truncated,
            notes=notes,
        )


def _truncate_middle(text: str, token_budget: int, count_tokens: TokenCounter) -> str:
    if count_tokens(text) <= token_budget:
        return text
    marker = "\n...[trimmed]...\n"
    # Binary search on characters: the token counter is the authority, not len().
    low, high = 0, len(text) // 2
    best = text[:0] + marker + text[len(text) :]
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid] + marker + text[len(text) - mid :]
        if count_tokens(candidate) <= token_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


class Conversation:
    """One session's history."""

    def __init__(
        self,
        *,
        session_id: int | str = "default",
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 40,
    ):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.turns: list[Turn] = []

    def add(self, role: str, content: str, *, turn_id: int | None = None) -> Turn:
        turn = Turn(role=role, content=content, turn_id=turn_id)
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]
        return turn

    def add_user(self, content: str, **kwargs: Any) -> Turn:
        return self.add("user", content, **kwargs)

    def add_assistant(self, content: str, **kwargs: Any) -> Turn:
        return self.add("assistant", content, **kwargs)

    def messages(self) -> list[Message]:
        out: list[Message] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        out.extend(t.as_message() for t in self.turns)
        return out

    def rewind(self, count: int = 1) -> None:
        """Forget the last ``count`` turns (used by /forget)."""
        if count > 0:
            self.turns = self.turns[:-count] or []

    def clear(self) -> None:
        self.turns.clear()

    @classmethod
    def from_history(
        cls, history: Sequence[Message], *, session_id: int | str = "default", **kwargs: Any
    ) -> "Conversation":
        conversation = cls(session_id=session_id, **kwargs)
        for message in history:
            if message["role"] == "system":
                conversation.system_prompt = message["content"]
            else:
                conversation.add(message["role"], message["content"])
        return conversation
