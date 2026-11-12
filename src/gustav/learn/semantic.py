"""Comparing an answer with its reference, semantically.

The gate's weakest link is scoring free text. Word overlap cannot tell a
differently-worded right answer from a wrong one, which means a candidate adapter that
*rephrases* well is scored as if it had failed — and the gate then refuses a real
improvement.

This compares mean-pooled hidden states from the model already in VRAM instead. It is
still a blunt instrument, so it is used the same careful way: as a share over a minimum
number of items, with a threshold, and with the overlap scorer as the fallback whenever
the backend cannot produce embeddings at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

#: Cosine similarity above which two answers are "the same answer". Chosen to be
#: forgiving about wording and unforgiving about content; the gate's own thresholds and
#: item count are what stop a single item from deciding anything.
DEFAULT_MATCH = 0.72

Embedder = Callable[[Sequence[str]], list[list[float]]]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two already-normalised vectors, clamped to 0..1."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return max(0.0, min(1.0, dot))


class SemanticMatcher:
    """Caches vectors, so the same reference answer is embedded once per run."""

    def __init__(self, embed: Embedder, *, threshold: float = DEFAULT_MATCH):
        self.embed = embed
        self.threshold = threshold
        self._cache: dict[str, list[float]] = {}
        self.calls = 0
        self.failed: str | None = None

    def vectors(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            self.calls += 1
            for text, vector in zip(missing, self.embed(missing), strict=True):
                self._cache[text] = vector
        return [self._cache.get(text, []) for text in texts]

    def similarity(self, left: str, right: str) -> float:
        if not (left or "").strip() or not (right or "").strip():
            return 0.0
        first, second = self.vectors([left, right])
        return cosine(first, second)

    def matches(self, answer: str, expected: str) -> tuple[bool, float]:
        score = self.similarity(answer, expected)
        return score >= self.threshold, score


def matcher_for(model: Any, *, threshold: float = DEFAULT_MATCH) -> SemanticMatcher | None:
    """A matcher backed by a loaded model, or None if it cannot embed.

    Never raises: a backend without embeddings is a reason to score with word overlap,
    not a reason to fail the night.
    """
    embed = getattr(model, "embed_texts", None)
    if embed is None:
        return None
    try:
        probe = embed(["probe"])
    except Exception:  # noqa: BLE001 - no embeddings is a fallback, not an error
        return None
    if not probe or not probe[0]:
        return None
    matcher = SemanticMatcher(embed, threshold=threshold)
    return matcher
