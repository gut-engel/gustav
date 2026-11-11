"""Topics, derived from the module registry.

A "topic" here is one of the keywords a module declares in ``modules/registry.yaml``.
That is a deliberate limitation rather than a shortcut: the registry is the vocabulary
the system has for talking about what it can do, the router already matches against it,
and self-expansion's job (stage 6) is to extend it. Inventing a parallel taxonomy would
give the skill table topics the router could never route to.
"""

from __future__ import annotations

from dataclasses import dataclass

from gustav.inference.registry import Registry
from gustav.inference.router import tokenize


@dataclass(frozen=True)
class Topic:
    module: str
    topic: str
    hits: int = 1

    def __str__(self) -> str:
        return f"{self.module}/{self.topic}"


def detect(text: str, registry: Registry, *, limit: int = 3) -> list[Topic]:
    """Which (module, topic) pairs this text is about, most specific first."""
    tokens = set(tokenize(text))
    if not tokens:
        return []
    found: list[Topic] = []
    for module in registry:
        hit_words = [k for k in module.keywords if k.lower() in tokens]
        if not hit_words:
            continue
        # A module that matched on several of its keywords is a better description of
        # the text than one that matched on a single generic word like "write".
        for word in hit_words:
            found.append(Topic(module=module.name, topic=word.lower(), hits=len(hit_words)))
    found.sort(key=lambda t: (-t.hits, t.module, t.topic))
    return found[:limit]


def parse(label: str) -> Topic:
    """``"rust/lifetime"`` -> Topic."""
    module, _, topic = label.partition("/")
    return Topic(module=module or "general", topic=topic or module)
