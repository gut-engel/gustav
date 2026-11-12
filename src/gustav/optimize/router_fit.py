"""Training the router on what the modules have actually been used for.

Stage 3 shipped a keyword scorer: IDF-weighted overlap between the request and each
module's hand-written description. It is cheap and explainable, and its limitation is
exactly the hand-written part — the rust module's keyword list does not contain
``borrowck``, ``Arc`` or ``tokio`` until somebody adds them, so a question using those
words and no listed keyword scores zero and is recorded as uncovered.

This fits a scorer from the material each module has accumulated instead: the prompts
of its training examples, its tasks, its benchmark items, plus its registry description
as an anchor. The weights are TF-IDF over *modules*, so a word that only ever appears in
one module's material becomes a strong signal for it and a word that appears everywhere
becomes noise.

Two honest caveats, which is why this installs only behind a check:

* **The labels are not independent of the old router.** A training example's module was
  assigned by the keyword scorer, so what this learns is "the vocabulary that shows up
  around the words the keyword list already knew" — a widening, not a correction. Real
  outcome-labelled routing would need per-module credit for an answer, which the system
  does not yet measure.
* **A bad fit is worse than no fit.** So a new model is installed only if it still
  agrees with the keyword scorer on the requests that scorer was confident about, and
  its threshold is calibrated so that the share of requests it calls *uncovered* does
  not suddenly jump — which would otherwise flood self-expansion with false triggers.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gustav.inference.registry import ModuleSpec, Registry
from gustav.inference.router import KeywordScorer, tokenize
from gustav.store.db import Database

#: How many times a module's registry description counts, so a module with no material
#: yet still scores for the words it was described with.
DESCRIPTION_WEIGHT = 3
#: Tokens kept per module, strongest first. Keeps the model small enough to read.
MAX_TOKENS_PER_MODULE = 400
#: Unknown words still count against a score, the way an unmatched IDF term does.
UNKNOWN_WEIGHT = 0.25

MODEL_FILE = "model.json"


@dataclass
class RouterModel:
    """A fitted scorer. Implements the ``Scorer`` protocol the router already takes."""

    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    threshold: float = 0.45
    modules: list[str] = field(default_factory=list)
    documents: dict[str, int] = field(default_factory=dict)
    trained_at: str = ""
    agreement: float | None = None
    uncovered_before: float | None = None
    uncovered_after: float | None = None

    # --- the Scorer protocol -------------------------------------------
    def score(self, text: str, modules: tuple[ModuleSpec, ...]) -> dict[str, float]:
        counts = Counter(tokenize(text))
        if not counts:
            return {m.name: 0.0 for m in modules}
        best = {
            token: max(
                (self.weights.get(m.name, {}).get(token, 0.0) for m in modules),
                default=0.0,
            )
            for token in counts
        }
        total = sum(
            count * max(best[token], UNKNOWN_WEIGHT) for token, count in counts.items()
        ) or 1.0
        out: dict[str, float] = {}
        for module in modules:
            table = self.weights.get(module.name, {})
            hit = sum(table.get(token, 0.0) * count for token, count in counts.items())
            out[module.name] = round(min(hit / total, 1.0), 4)
        return out

    # --- storage --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "modules": self.modules,
            "documents": self.documents,
            "trained_at": self.trained_at,
            "agreement": self.agreement,
            "uncovered_before": self.uncovered_before,
            "uncovered_after": self.uncovered_after,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouterModel":
        return cls(
            weights={k: dict(v) for k, v in (data.get("weights") or {}).items()},
            threshold=float(data.get("threshold", 0.45)),
            modules=list(data.get("modules") or []),
            documents=dict(data.get("documents") or {}),
            trained_at=str(data.get("trained_at") or ""),
            agreement=data.get("agreement"),
            uncovered_before=data.get("uncovered_before"),
            uncovered_after=data.get("uncovered_after"),
        )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_FILE
        temporary = path.with_name(f".{MODEL_FILE}.part")
        temporary.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, directory: Path) -> "RouterModel | None":
        path = Path(directory) / MODEL_FILE
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None


def module_documents(db: Database, registry: Registry) -> dict[str, list[str]]:
    """The text each module has been used for, including its own description."""
    documents: dict[str, list[str]] = {
        module.name: [module.text] * DESCRIPTION_WEIGHT for module in registry
    }
    queries = (
        "SELECT module, prompt AS text FROM training_examples",
        "SELECT module, prompt AS text FROM tasks",
        "SELECT module, prompt AS text FROM benchmark_items WHERE retired = 0",
        "SELECT module, topic AS text FROM skills WHERE demand > 0",
    )
    for sql in queries:
        for row in db.query(sql):
            if row["module"] in documents and row["text"]:
                documents[row["module"]].append(str(row["text"]))
    return documents


def fit_weights(documents: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """How strongly each word indicates each module.

    ``p(t | m) * max(p(t | m) / p(t) - 1, 0)``, normalised so a module's strongest word
    scores 1.0. The two factors do different jobs: the first prefers words the module
    actually uses a lot, the second prefers words *only* it uses — and because it is a
    ratio against the global frequency, a word that appears equally everywhere comes
    out at exactly zero rather than merely small.

    Plain TF-IDF is not enough here. With a handful of modules its IDF term barely
    separates anything, so a frequent pleasantry can outweigh a rare technical term.
    """
    per_module: dict[str, Counter[str]] = {}
    for module, texts in documents.items():
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        per_module[module] = counts

    global_counts: Counter[str] = Counter()
    for counts in per_module.values():
        global_counts.update(counts)
    global_total = sum(global_counts.values()) or 1

    weights: dict[str, dict[str, float]] = {}
    for module, counts in per_module.items():
        total = sum(counts.values()) or 1
        scored: dict[str, float] = {}
        for token, count in counts.items():
            in_module = count / total
            overall = global_counts[token] / global_total
            ratio = in_module / overall if overall else 0.0
            distinctiveness = max(ratio - 1.0, 0.0)
            value = in_module * distinctiveness
            if value > 0:
                scored[token] = value
        if not scored:
            weights[module] = {}
            continue
        peak = max(scored.values()) or 1.0
        trimmed = sorted(scored.items(), key=lambda kv: -kv[1])[:MAX_TOKENS_PER_MODULE]
        weights[module] = {token: round(value / peak, 4) for token, value in trimmed}
    return weights


def recent_requests(db: Database, *, limit: int = 300) -> list[str]:
    rows = db.query(
        "SELECT content FROM turns WHERE role = 'user' ORDER BY id DESC LIMIT ?", (limit,)
    )
    return [row["content"] for row in rows if row["content"]]


def calibrate(
    model: RouterModel,
    requests: list[str],
    registry: Registry,
    *,
    keyword_threshold: float,
    floor: float = 0.1,
) -> tuple[float, float, float]:
    """Pick a threshold that keeps the 'uncovered' rate where it was.

    Returns ``(threshold, uncovered_before, uncovered_after)``. With no history the
    keyword threshold is kept: changing it on no evidence would be the opposite of the
    point.
    """
    modules = registry.modules
    if not requests or not modules:
        return keyword_threshold, 0.0, 0.0

    keyword = KeywordScorer(modules)
    before_scores = [max(keyword.score(text, modules).values(), default=0.0) for text in requests]
    after_scores = [max(model.score(text, modules).values(), default=0.0) for text in requests]
    uncovered_before = sum(s < keyword_threshold for s in before_scores) / len(requests)

    # The threshold that reproduces the old uncovered rate under the new scorer.
    ranked = sorted(after_scores)
    index = min(int(uncovered_before * len(ranked)), len(ranked) - 1)
    threshold = max(round(ranked[index], 4), floor) if ranked else keyword_threshold
    uncovered_after = sum(s < threshold for s in after_scores) / len(requests)
    return threshold, round(uncovered_before, 4), round(uncovered_after, 4)


def agreement_with_keywords(
    model: RouterModel, requests: list[str], registry: Registry, *, keyword_threshold: float
) -> float | None:
    """How often the fitted scorer's best module matches the keyword scorer's.

    Only on requests the keyword scorer was confident about — it has no opinion worth
    agreeing with on the others, which are the ones this model exists to improve.
    """
    modules = tuple(m for m in registry.modules if not m.always_on)
    if not modules or not requests:
        return None
    keyword = KeywordScorer(registry.modules)
    agree = 0
    considered = 0
    for text in requests:
        keyword_scores = keyword.score(text, modules)
        best_keyword = max(keyword_scores.items(), key=lambda kv: kv[1], default=None)
        if best_keyword is None or best_keyword[1] < keyword_threshold:
            continue
        considered += 1
        learned_scores = model.score(text, modules)
        best_learned = max(learned_scores.items(), key=lambda kv: kv[1], default=None)
        if best_learned and best_learned[0] == best_keyword[0]:
            agree += 1
    if not considered:
        return None
    return round(agree / considered, 4)


def fit(
    db: Database,
    registry: Registry,
    *,
    keyword_threshold: float,
    min_documents: int = 20,
) -> tuple[RouterModel | None, str]:
    """Fit a router from the store. Returns ``(model, reason)``; model is None if not fit."""
    documents = module_documents(db, registry)
    total = sum(len(texts) for texts in documents.values())
    material = total - DESCRIPTION_WEIGHT * len(documents)
    if material < min_documents:
        return None, (
            f"only {material} pieces of module material; {min_documents} are needed "
            f"before a fitted router would be anything but the keyword list again"
        )

    weights = fit_weights(documents)
    model = RouterModel(
        weights=weights,
        modules=registry.names,
        documents={name: len(texts) for name, texts in documents.items()},
        trained_at=datetime.now().isoformat(timespec="seconds"),
    )
    requests = recent_requests(db)
    model.threshold, model.uncovered_before, model.uncovered_after = calibrate(
        model, requests, registry, keyword_threshold=keyword_threshold
    )
    model.agreement = agreement_with_keywords(
        model, requests, registry, keyword_threshold=keyword_threshold
    )
    return model, f"fitted on {material} pieces of material from {len(documents)} modules"
