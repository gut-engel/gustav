"""HTML to text, with the structure that matters for study kept.

Documentation pages are mostly navigation. What study needs is the prose and the code
blocks, in order, with headings intact so a chunk can say what it is about — and it
needs that without adding a dependency that parses the whole DOM.

``html.parser`` is in the standard library and tolerant of the malformed markup real
pages contain, which is why this is a parser rather than a pile of regexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

#: Elements whose content is never prose.
#: `head` is not here: its `<title>` is the page's name, and everything else it holds
#: either produces no text or is dropped by name anyway.
_DROP = frozenset({"script", "style", "noscript", "template", "svg", "nav",
                   "footer", "form", "button", "select"})
_BLOCK = frozenset({"p", "div", "section", "article", "li", "tr", "br", "hr", "blockquote",
                    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "table", "dd", "dt"})
_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


@dataclass
class Document:
    title: str = ""
    text: str = ""
    headings: list[str] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.headings: list[str] = []
        self.code_blocks: list[str] = []
        self._drop_depth = 0
        self._in_title = False
        self._in_pre = False
        self._pre: list[str] = []
        self._heading: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "pre":
            self._in_pre = True
            self._pre = []
            self.parts.append("\n")
        elif tag in _HEADINGS:
            self._heading = []
            self.parts.append("\n\n")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if self._drop_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "pre":
            block = "".join(self._pre).strip("\n")
            if block.strip():
                self.code_blocks.append(block)
                self.parts.append(f"\n```\n{block}\n```\n")
            self._in_pre = False
            self._pre = []
        elif tag in _HEADINGS:
            heading = _WS.sub(" ", "".join(self._heading or []).strip())
            if heading:
                self.headings.append(heading)
                self.parts.append(f"\n{'#' * int(tag[1])} {heading}\n")
            self._heading = None
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        if self._in_pre:
            self._pre.append(data)
            return
        if self._heading is not None:
            self._heading.append(data)
            return
        self.parts.append(_WS.sub(" ", data))


def to_text(html: str, *, title: str = "") -> Document:
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup is not an error worth a night
        pass
    text = _BLANKS.sub("\n\n", "".join(parser.parts))
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return Document(
        title=(parser.title or title).strip(),
        text=text,
        headings=parser.headings,
        code_blocks=parser.code_blocks,
    )


def looks_like_html(data: str) -> bool:
    head = data[:2048].lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head


def read(data: str, *, title: str = "") -> Document:
    """Text or HTML, whichever it turns out to be."""
    if looks_like_html(data):
        return to_text(data, title=title)
    headings = [
        line.lstrip("# ").strip()
        for line in data.splitlines()
        if line.startswith("#") and len(line) < 120
    ]
    blocks = re.findall(r"```[\w+-]*\n(.*?)```", data, re.DOTALL)
    return Document(
        title=title or (headings[0] if headings else ""),
        text=data.strip(),
        headings=headings,
        code_blocks=[b.strip() for b in blocks],
    )
