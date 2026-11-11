"""Reading structure out of model output.

A 4B model asked for JSON returns JSON *most* of the time: sometimes in a fenced block,
sometimes after a sentence of introduction, sometimes with a trailing comma. Rather than
re-prompting until it complies — expensive, and it does not always converge — the night
parses leniently and discards whatever still will not parse.

Nothing here makes the content trustworthy. It makes it *typed*; the caller still has to
check that a fact is grounded or that a solution passes its tests.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|jsonl|python)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")


def _candidates(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    out = [block.strip() for block in _FENCE.findall(text)]
    out.append(text)
    # The outermost array or object, for an answer wrapped in prose.
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            out.append(text[start : end + 1])
    return out


def loads(text: str) -> Any:
    """Parse the most JSON-looking part of ``text``, or return None."""
    for candidate in _candidates(text):
        for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def objects(text: str) -> list[dict[str, Any]]:
    """Every object in the answer, whether it came as an array, one object, or JSONL."""
    parsed = loads(text)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    found: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            item = loads(line)
            if isinstance(item, dict):
                found.append(item)
    return found


def strings(text: str) -> list[str]:
    """A list of strings, from JSON, from JSONL, or from a bulleted list."""
    parsed = loads(text)
    if isinstance(parsed, list):
        items = [str(i).strip() for i in parsed if isinstance(i, (str, int, float))]
        if items:
            return items
    found: list[str] = []
    for line in (text or "").splitlines():
        match = _BULLET.match(line)
        if match:
            found.append(match.group(1).strip().strip('"'))
    return found
