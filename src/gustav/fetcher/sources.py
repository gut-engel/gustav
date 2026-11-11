"""Which documentation a topic maps to.

There is no search engine here, and that is the design. A topic is looked up in
``docs/sources.yaml`` or found in a local directory, which means the set of things the
daemon can ever fetch is a file you can read and edit — not a query whose results
depend on what a search index felt like returning tonight.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

#: Files a local source directory is allowed to offer.
LOCAL_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".html")
_WORD = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class SourceEntry:
    urls: tuple[str, ...]
    module: str | None = None
    topics: tuple[str, ...] = ()
    #: The licence of the material, as *you* recorded it. Nothing verifies this; it is
    #: here so that "may I train on this" is answerable afterwards.
    license: str | None = None

    def matches(self, module: str, topic: str) -> int:
        """0 = no match, 1 = the module's fallback reading, 2 = this exact topic."""
        if self.module and self.module != module:
            return 0
        if not self.topics:
            return 1
        return 2 if topic.lower() in self.topics else 0


@dataclass(frozen=True)
class Candidate:
    url: str
    module: str
    topic: str
    #: A local file, read straight from disk rather than fetched.
    path: Path | None = None
    score: int = 1
    license: str | None = None

    @property
    def local(self) -> bool:
        return self.path is not None

    @property
    def host(self) -> str:
        return urlparse(self.url).hostname or ""


@dataclass
class Sources:
    entries: tuple[SourceEntry, ...] = ()
    local_dirs: tuple[Path, ...] = ()
    path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def for_topic(self, module: str, topic: str) -> list[Candidate]:
        found: list[Candidate] = []
        for entry in self.entries:
            score = entry.matches(module, topic)
            if not score:
                continue
            found.extend(
                Candidate(
                    url=url, module=module, topic=topic, score=score, license=entry.license
                )
                for url in entry.urls
            )
        found.extend(self._local_for(module, topic))
        # Topic-specific sources first, and a stable order after that so two nights
        # with the same topics do not fetch in a different order for no reason.
        found.sort(key=lambda c: (-c.score, c.url))
        return found

    def _local_for(self, module: str, topic: str) -> list[Candidate]:
        found: list[Candidate] = []
        needle = topic.lower()
        for directory in self.local_dirs:
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in LOCAL_SUFFIXES:
                    continue
                score = 0
                if needle in path.stem.lower():
                    score = 2
                else:
                    try:
                        head = path.read_text(encoding="utf-8", errors="replace")[:8192]
                    except OSError:
                        continue
                    if needle in {w for w in _WORD.findall(head.lower())}:
                        score = 1
                if score:
                    found.append(
                        Candidate(
                            url=path.resolve().as_uri(),
                            module=module,
                            topic=topic,
                            path=path,
                            score=score + 1,  # local material beats a network fetch
                            license="local",  # your own files, your own call
                        )
                    )
        return found

    def candidates(
        self, topics: Iterable[tuple[str, str]], *, limit: int = 12
    ) -> list[Candidate]:
        """Candidates for several topics, interleaved so one topic cannot eat the budget."""
        per_topic = [self.for_topic(module, topic) for module, topic in topics]
        out: list[Candidate] = []
        seen: set[str] = set()
        index = 0
        while len(out) < limit and any(index < len(group) for group in per_topic):
            for group in per_topic:
                if index >= len(group) or len(out) >= limit:
                    continue
                candidate = group[index]
                if candidate.url in seen:
                    continue
                seen.add(candidate.url)
                out.append(candidate)
            index += 1
        return out


def load_sources(
    path: Path | str | None, *, local_dirs: Sequence[Path | str] = ()
) -> Sources:
    entries: list[SourceEntry] = []
    warnings: list[str] = []
    source_path = Path(path) if path else None
    if source_path and source_path.exists():
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        for item in raw.get("sources") or []:
            urls = tuple(str(u).strip() for u in (item.get("urls") or []) if str(u).strip())
            if not urls:
                warnings.append(f"{source_path}: an entry has no urls and was ignored")
                continue
            entries.append(
                SourceEntry(
                    urls=urls,
                    module=(item.get("module") or None),
                    topics=tuple(str(t).lower() for t in (item.get("topics") or [])),
                    license=(item.get("license") or None),
                )
            )
    return Sources(
        entries=tuple(entries),
        local_dirs=tuple(Path(d).expanduser() for d in local_dirs),
        path=source_path,
        warnings=warnings,
    )
