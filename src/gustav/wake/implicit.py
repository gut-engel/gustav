"""Implicit feedback.

When a user rephrases the same question, or says "no, that's wrong", that is a signal —
but a weak and ambiguous one. So it is recorded as a *candidate* for review and never
trained on automatically. ``ai facts``/the report surface it; only a human marks it
reviewed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Phrases that usually mean the previous answer missed.
NEGATIVE_PHRASES = (
    "that's wrong", "thats wrong", "that is wrong", "you're wrong", "youre wrong",
    "not what i asked", "not what i meant", "no, i meant", "no i meant",
    "that doesn't work", "that does not work", "that didn't work", "that did not work",
    "doesn't compile", "does not compile", "didn't compile", "did not compile",
    "doesn't build", "didn't build", "still broken", "still fails", "still wrong",
    "that's not what", "that is not what", "not quite", "you're missing",
    "you misunderstood", "you missed", "wrong answer", "that's incorrect",
    "that is incorrect", "nonsense", "makes no sense", "try again", "no that's not",
    "incorrect", "not true", "hallucinat",
)
#: Phrases that usually mean it landed.
POSITIVE_PHRASES = (
    "thanks", "thank you", "that worked", "works now", "perfect", "exactly",
    "that's right", "that is right", "nice one", "great, that", "that fixed it",
)

_WORD = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from how i if in is it its me my "
    "no not of on or please should so that the their there they this to was what when "
    "where which who why will with would you your".split()
)


@dataclass(frozen=True)
class ImplicitSignal:
    kind: str           # "negative" | "rephrase" | "positive"
    confidence: float   # 0..1, deliberately never 1.0
    reason: str

    @property
    def negative(self) -> bool:
        return self.kind in ("negative", "rephrase")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def similarity(left: str, right: str) -> float:
    """Jaccard overlap of content words. Cheap, and good enough to spot a rephrase."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect(
    message: str,
    *,
    previous_user: str | None = None,
    rephrase_threshold: float = 0.55,
) -> ImplicitSignal | None:
    """Look at one new user message in the light of the previous one."""
    lowered = message.strip().lower()
    if not lowered:
        return None

    for phrase in NEGATIVE_PHRASES:
        if phrase in lowered:
            return ImplicitSignal(
                kind="negative",
                confidence=0.6,
                reason=f"the user said {phrase!r}",
            )

    # A short "no" on its own, right after an answer.
    if lowered in {"no", "nope", "wrong", "nah"}:
        return ImplicitSignal(kind="negative", confidence=0.5, reason="a bare rejection")

    for phrase in POSITIVE_PHRASES:
        if phrase in lowered:
            return ImplicitSignal(
                kind="positive", confidence=0.5, reason=f"the user said {phrase!r}"
            )

    if previous_user:
        overlap = similarity(previous_user, message)
        if overlap >= rephrase_threshold and len(_tokens(message)) >= 3:
            # Asking almost the same thing again usually means the answer did not land.
            return ImplicitSignal(
                kind="rephrase",
                confidence=min(0.3 + overlap / 2, 0.75),
                reason=f"the question was rephrased ({overlap:.0%} word overlap)",
            )
    return None
