"""Deterministic fact extraction.

This runs whether or not a model is available, and it is the only extractor whose
output is trusted without review. The patterns are narrow on purpose: a false fact
about the user is worse than a missed one, because it will be trained on and then
repeated back with confidence.

Two rules shape everything here:

* **Only statements, never questions.** "my editor is helix" is a fact; "what is my
  editor?" is not, and the difference is one character, so it is checked explicitly.
* **Facts are rewritten into the third person.** Training on "my editor is helix" would
  teach the model to say *its* editor is helix. Stored facts read "the user's editor is
  helix", which is what we actually want it to learn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gustav.learn.facts import Fact

#: A value we refuse to store, however it was matched.
_REJECT = re.compile(
    r"\[REDACTED|password|secret|api[_ -]?key|token|ssh-rsa|BEGIN [A-Z ]*PRIVATE KEY",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^(what|who|where|when|why|how|which|do|does|did|is|are|was|were|can|could|should|"
    r"would|will|am|have|has|tell|show|list|remind)\b",
    re.IGNORECASE,
)
#: Split on sentence enders only when a space or the end follows, so "deploy.sh"
#: and "3.14" stay in one piece.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

MIN_VALUE = 2
MAX_VALUE = 160


@dataclass(frozen=True)
class Pattern:
    #: Matched against a single sentence, case-insensitively. Named groups ``value``
    #: and ``thing`` are used when present; otherwise the groups are read in order.
    regex: re.Pattern[str]
    #: The stable key a later statement supersedes. ``None`` means "never supersede":
    #: two preferences can both be true, two values of "name" cannot.
    subject: str | None
    #: ``{value}`` and, where the pattern has one, ``{thing}``.
    template: str
    confidence: float = 0.7


PATTERNS: tuple[Pattern, ...] = (
    Pattern(re.compile(r"\bmy name(?:'s| is) (?P<value>[\w .'-]{2,60})", re.I), "name",
            "The user's name is {value}.", 0.9),
    Pattern(re.compile(r"\b(?:call me|i'?m called) (?P<value>[\w .'-]{2,60})", re.I), "name",
            "The user's name is {value}.", 0.85),
    Pattern(re.compile(r"\b(?:i am|i'm) (?:a|an) (?P<value>[\w /+-]{2,60})", re.I), "role",
            "The user is a {value}.", 0.7),
    Pattern(re.compile(r"\bi (?:live|am based) in (?P<value>[\w .,'-]{2,60})", re.I),
            "location", "The user lives in {value}.", 0.8),
    Pattern(re.compile(r"\bi work (?:at|for) (?P<value>[\w .,'&-]{2,60})", re.I), "employer",
            "The user works at {value}.", 0.8),
    Pattern(re.compile(r"\bi work on (?P<value>[\w .,'&/-]{2,80})", re.I), "project",
            "The user works on {value}.", 0.7),
    Pattern(re.compile(r"\bi (?:use|run) (?P<value>[\w .,'+/-]{2,80}) for "
                       r"(?P<thing>[\w .,'+/-]{2,40})", re.I),
            "tool:{thing}", "The user uses {value} for {thing}.", 0.7),
    Pattern(re.compile(r"\bmy (?P<thing>[\w -]{2,30}) (?:is|are) (?P<value>[^!?\n]{2,120})",
                       re.I),
            "{thing}", "The user's {thing} is {value}.", 0.75),
    Pattern(re.compile(r"\bi (?:prefer|always want|'d rather have) (?P<value>[^!?\n]{3,120})",
                       re.I), None, "The user prefers {value}.", 0.7),
    Pattern(re.compile(r"\bi (?:hate|dislike|never want|don'?t want) "
                       r"(?P<value>[^!?\n]{3,120})", re.I), None,
            "The user does not want {value}.", 0.7),
    Pattern(re.compile(r"\b(?:remember|note|keep in mind) (?:that )?(?P<value>[^!?\n]{4,160})",
                       re.I), None, "{value}", 0.85),
)

_FIRST_PERSON = (
    (re.compile(r"\bmy\b", re.IGNORECASE), "the user's"),
    (re.compile(r"\bi am\b", re.IGNORECASE), "the user is"),
    (re.compile(r"\bi'm\b", re.IGNORECASE), "the user is"),
    (re.compile(r"\bi\b", re.IGNORECASE), "the user"),
    (re.compile(r"\bme\b", re.IGNORECASE), "the user"),
    (re.compile(r"\bmine\b", re.IGNORECASE), "the user's"),
)


def depersonalise(text: str) -> str:
    """Rewrite first person into statements about the user."""
    out = text
    for pattern, replacement in _FIRST_PERSON:
        out = pattern.sub(replacement, out)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


#: Conversational filler that is not part of the value.
_FILLER = re.compile(
    r",?\s*\b(?:btw|by the way|though|anyway|i think|i guess|you know|right\?*)\s*$",
    re.IGNORECASE,
)


def _clean(value: str) -> str:
    value = value.strip().strip("\"'`")
    value = re.sub(r"\s+", " ", value)
    value = _FILLER.sub("", value)
    return value.rstrip(" ,;:.")


def _acceptable(value: str) -> bool:
    if not (MIN_VALUE <= len(value) <= MAX_VALUE):
        return False
    if _REJECT.search(value):
        return False
    # A value that is itself a question or an instruction is a parse failure.
    return not _QUESTION_START.match(value)


def facts_from_message(
    text: str, *, turn_id: int | None = None, module: str = "personal-memory"
) -> list[Fact]:
    """Facts stated in one user message. Empty is the normal case."""
    found: list[Fact] = []
    seen: set[str] = set()
    for sentence in sentences(text):
        if _QUESTION_START.match(sentence) or "?" in sentence:
            continue
        for pattern in PATTERNS:
            match = pattern.regex.search(sentence)
            if not match:
                continue
            named = match.groupdict()
            raw_value = named.get("value")
            raw_thing = named.get("thing") or ""
            if raw_value is None:
                continue
            thing, value = _clean(raw_thing), _clean(raw_value)
            if "{thing}" in pattern.template and not thing:
                continue
            if not _acceptable(value) or (thing and not _acceptable(thing)):
                continue
            body = pattern.template.format(value=depersonalise(value), thing=thing).strip()
            body = body[0].upper() + body[1:] if body else body
            if not body.endswith("."):
                body += "."
            if body.lower() in seen:
                continue
            seen.add(body.lower())
            # Subjects are the supersede key, so they are normalised: "GPU" and "gpu"
            # are the same thing being restated, not two facts.
            subject = (
                pattern.subject.format(thing=thing).lower() if pattern.subject else None
            )
            found.append(
                Fact(
                    text=body,
                    subject=subject,
                    module=module,
                    confidence=pattern.confidence,
                    source="turn",
                    source_turn_id=turn_id,
                )
            )
            break  # one fact per sentence: the most specific pattern that matched
    return found


def question_for(fact: Fact) -> str:
    """A prompt that should elicit a fact, for the personal-memory training set."""
    if fact.subject and not fact.subject.startswith("tool:"):
        return f"What do you know about my {fact.subject.replace('_', ' ')}?"
    return "What do you know about me that is relevant here?"
