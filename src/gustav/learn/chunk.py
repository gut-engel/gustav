"""Splitting a document into pieces worth studying.

Headings are the natural boundary: a section is about one thing, which is what makes a
citation meaningful ("the lifetime elision section of the Rust book says ..."). Inside a
section, paragraphs are packed up to a character budget rather than cut at it, because a
chunk that ends mid-sentence produces a study question with no answer in it.

Code blocks are kept whole even when they blow the budget. A truncated example is worse
than no example: it will not compile, and something will eventually try to run it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^```")

DEFAULT_MAX = 1400
DEFAULT_MIN = 180


@dataclass(frozen=True)
class Chunk:
    text: str
    heading: str
    position: int

    @property
    def has_code(self) -> bool:
        return "```" in self.text

    def citation(self, url: str) -> str:
        if not self.heading:
            return url
        anchor = re.sub(r"[^a-z0-9]+", "-", self.heading.lower()).strip("-")
        return f"{url}#{anchor}" if anchor else url


def _blocks(text: str) -> list[tuple[str, str]]:
    """(heading, block) pairs: paragraphs and fenced code, in order."""
    out: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            out.append((heading, body))
        buffer.clear()

    for line in (text or "").splitlines():
        if _FENCE.match(line.strip()):
            if in_code:
                buffer.append(line)
                in_code = False
                flush()
            else:
                flush()
                in_code = True
                buffer.append(line)
            continue
        if in_code:
            buffer.append(line)
            continue
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group(2)
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return out


def chunks(
    text: str, *, max_chars: int = DEFAULT_MAX, min_chars: int = DEFAULT_MIN
) -> list[Chunk]:
    found: list[Chunk] = []
    current_heading = ""
    buffer: list[str] = []
    size = 0

    def emit() -> None:
        nonlocal buffer, size
        body = "\n\n".join(buffer).strip()
        buffer, size = [], 0
        if not body:
            return
        if len(body) < min_chars and "```" not in body:
            return
        found.append(Chunk(text=body, heading=current_heading, position=len(found)))

    for heading, block in _blocks(text):
        if heading != current_heading:
            emit()
            current_heading = heading
        if size and size + len(block) > max_chars:
            emit()
        buffer.append(block)
        size += len(block) + 2
        if size >= max_chars:
            emit()
    emit()
    return found


def best_chunks(items: list[Chunk], topic: str, *, limit: int = 4) -> list[Chunk]:
    """The chunks most likely to be about the topic, in document order."""
    needle = (topic or "").lower()

    def relevance(chunk: Chunk) -> tuple[int, int]:
        body = chunk.text.lower()
        return (
            body.count(needle) + 2 * (needle in chunk.heading.lower()) if needle else 0,
            len(chunk.text),
        )

    ranked = sorted(items, key=relevance, reverse=True)[:limit]
    return sorted(ranked, key=lambda c: c.position)
