"""Is this statement actually supported by that text?

A cheap content-word overlap test, used wherever the model is asked to read something
and say what it found: extracted facts, study answers, summaries. It cannot catch a
subtle distortion. It does catch the common and dangerous failure — a fluent, plausible
claim that nothing in the source supports — which is the one that otherwise ends up in
the training set and then gets asserted with confidence.
"""

from __future__ import annotations

from gustav.inference.router import tokenize
from gustav.learn.extract import depersonalise

#: Share of a claim's content words that must appear in the source.
DEFAULT_THRESHOLD = 0.5

#: Words that carry no claim, so their presence or absence proves nothing.
_FILLER = frozenset({"user", "users", "also", "well", "thing", "things", "use", "used",
                     "using", "example", "code", "value", "values"})


def _words(text: str) -> set[str]:
    """Content words, with punctuation stripped.

    The router's tokenizer keeps dots and dashes inside a token, which is right for
    matching `tokio-stream` or `v1.2` against a module description. Here it would make
    "answers." and "answers" two different words, which is exactly the comparison this
    function exists to get right.
    """
    return {
        word
        for raw in tokenize(depersonalise(text))
        if (word := raw.strip(".-_")) and word not in _FILLER
    }


def grounded(statement: str, source: str, *, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return overlap(statement, source) >= threshold


def overlap(statement: str, source: str) -> float:
    """Share of the statement's content words that appear in the source."""
    claim = _words(statement)
    if not claim:
        return 0.0
    return len(claim & _words(source)) / len(claim)
