"""Getting runnable code out of a chat answer.

A model asked for a function replies with a function, usually inside a fenced block,
often with a sentence before it and an explanation after. Both have to go: the sandbox
runs what comes out of here, and prose at module level is a syntax error.
"""

from __future__ import annotations

import re

_FENCED = re.compile(r"```(?:py|python|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```\w*\s*\n(.*?)```", re.DOTALL)
#: A line that starts a code construct. Used to find where prose stops.
_CODE_START = re.compile(r"^(?:from |import |def |class |async def |@|#|[A-Za-z_]\w*\s*=)")


def parses(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def extract_python(text: str) -> str:
    """The code in an answer, or "" if there is none that parses."""
    if not text:
        return ""
    blocks = _FENCED.findall(text) or _ANY_FENCE.findall(text)
    for block in blocks:
        candidate = block.strip()
        if candidate and parses(candidate):
            return candidate
    # Unfenced: drop leading prose, then trailing prose, until it compiles.
    lines = text.strip().splitlines()
    start = next((i for i, line in enumerate(lines) if _CODE_START.match(line)), None)
    if start is None:
        return ""
    body = lines[start:]
    while body:
        candidate = "\n".join(body).strip()
        if candidate and parses(candidate):
            return candidate
        body = body[:-1]
    return ""
