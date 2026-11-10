"""The router: which adapters does this request need?

Runs *before* generation, so the right modules are resident when the first token is
produced. Stage 3 ships a keyword/overlap scorer, which is cheap and explainable; the
``Scorer`` protocol is where an embedding model drops in, and sleep's stage 8 trains the
weights from logged decisions and their outcomes instead of fixed thresholds.

Every decision is logged with its scores, which is what makes "train the router on
logged routing decisions" possible later.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gustav.inference.registry import ModuleSpec, Registry

_WORD = re.compile(r"[a-z0-9_+#.-]+")
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does for from had has have how i if "
    "in into is it its me my of on or should so than that the their them then there "
    "these they this to was were what when where which who why will with would you your"
    .split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


@runtime_checkable
class Scorer(Protocol):
    """Scores a request against each module. Returns name -> score in [0, 1]."""

    def score(self, text: str, modules: tuple[ModuleSpec, ...]) -> dict[str, float]: ...


class KeywordScorer:
    """Inverse-document-frequency weighted keyword overlap.

    A word that appears in every module's description says nothing; a word that appears
    in one says a lot. That is all IDF is, and it is enough to tell "borrow checker"
    from "summarise this email" without loading an embedding model.
    """

    def __init__(self, modules: tuple[ModuleSpec, ...]):
        self.modules = modules
        self._vocab: dict[str, set[str]] = {}
        for module in modules:
            for token in set(tokenize(module.text)):
                self._vocab.setdefault(token, set()).add(module.name)
        self._count = max(len(modules), 1)

    def _idf(self, token: str) -> float:
        documents = len(self._vocab.get(token, ()))
        if documents == 0:
            return 0.0
        return math.log((1 + self._count) / (1 + documents)) + 1.0

    def score(self, text: str, modules: tuple[ModuleSpec, ...]) -> dict[str, float]:
        tokens = Counter(tokenize(text))
        if not tokens:
            return {m.name: 0.0 for m in modules}
        scores: dict[str, float] = {}
        total = sum(self._idf(t) * c for t, c in tokens.items()) or 1.0
        for module in modules:
            module_tokens = set(tokenize(module.text))
            hit = sum(
                self._idf(token) * count
                for token, count in tokens.items()
                if token in module_tokens
            )
            scores[module.name] = min(hit / total, 1.0)
        return scores


@dataclass
class RouteDecision:
    modules: tuple[str, ...]
    scores: dict[str, float]
    threshold: float
    reason: str
    confidence: float = 0.0
    #: Modules that scored but were cut by max_active.
    dropped: tuple[str, ...] = ()
    #: True when nothing scored above the threshold — a signal that no module covers
    #: this topic, which is one of the triggers for self-expansion (section 8).
    uncovered: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        ranked = ", ".join(
            f"{name} {score:.2f}" for name, score in sorted(
                self.scores.items(), key=lambda kv: -kv[1]
            )[:4]
        )
        return f"{list(self.modules)} (threshold {self.threshold:.2f}; {ranked})"


class Router:
    def __init__(
        self,
        registry: Registry,
        *,
        scorer: Scorer | None = None,
        threshold: float | None = None,
        max_active: int | None = None,
        available: set[str] | None = None,
    ):
        self.registry = registry
        self.scorer = scorer or KeywordScorer(registry.modules)
        self.threshold = registry.threshold if threshold is None else threshold
        self.max_active = registry.max_active if max_active is None else max_active
        #: Modules that actually have trained weights. Unknown means "all of them".
        self.available = available

    def _usable(self) -> tuple[ModuleSpec, ...]:
        if self.available is None:
            return self.registry.modules
        return tuple(m for m in self.registry.modules if m.name in self.available)

    def route(self, text: str, *, forced: tuple[str, ...] = ()) -> RouteDecision:
        usable = self._usable()
        if forced:
            known = tuple(f for f in forced if f in self.registry)
            return RouteDecision(
                modules=known,
                scores={name: 1.0 for name in known},
                threshold=self.threshold,
                confidence=1.0,
                reason="modules were requested explicitly",
            )
        if not usable:
            return RouteDecision(
                modules=(), scores={}, threshold=self.threshold,
                reason="no trained modules are available yet",
            )

        scores = self.scorer.score(text, usable)
        always = [m for m in usable if m.always_on]
        chosen = {m.name for m in always}
        over = [
            m for m in usable
            if m.name not in chosen and scores.get(m.name, 0.0) >= self.threshold
        ]
        over.sort(key=lambda m: (-scores.get(m.name, 0.0), -m.priority))

        dropped: list[str] = []
        for module in over:
            if len(chosen) < self.max_active:
                chosen.add(module.name)
            else:
                dropped.append(module.name)

        ranked = sorted(
            (m for m in usable if m.name in chosen),
            key=lambda m: (-m.priority, -scores.get(m.name, 0.0)),
        )
        # Confidence is the best score among the modules that had to earn their place.
        best_specific = max(
            (scores.get(m.name, 0.0) for m in usable if not m.always_on), default=0.0
        )
        # "Uncovered" means no module handles this topic at all — the expansion trigger
        # from section 8. An always-on module scoring well still counts as coverage, so
        # an everyday question is not mistaken for a gap in the model's skills.
        best_overall = max(scores.values(), default=0.0)
        uncovered = best_overall < self.threshold
        reason = (
            f"{len(over)} module(s) above the {self.threshold:.2f} threshold"
            if over
            else f"nothing above the {self.threshold:.2f} threshold; always-on modules only"
        )
        return RouteDecision(
            modules=tuple(m.name for m in ranked),
            scores=scores,
            threshold=self.threshold,
            confidence=round(best_specific, 4),
            reason=reason,
            dropped=tuple(dropped),
            uncovered=uncovered,
        )


class RouterLog:
    """Where routing decisions go, so they can be learned from later."""

    def __init__(self, db=None):
        self.db = db

    def record(self, decision: RouteDecision, *, turn_id: int | None = None) -> None:
        if self.db is None:
            return
        import json

        self.db.insert(
            "telemetry",
            {
                "metric": "router.decision",
                "value": decision.confidence,
                "tags": json.dumps(
                    {
                        "modules": list(decision.modules),
                        "scores": {k: round(v, 4) for k, v in decision.scores.items()},
                        "threshold": decision.threshold,
                        "uncovered": decision.uncovered,
                        "turn_id": turn_id,
                    }
                ),
            },
        )
