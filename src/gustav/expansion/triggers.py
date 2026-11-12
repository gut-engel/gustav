"""The measurements that justify growing.

Section 8 of the specification says capacity may grow *only if measurements demand it*,
so this module contains no policy at all — it reads the store and reports what it finds.
Deciding what to do about a trigger, and whether the budget allows it, is
``proposals.py``; doing it is ``actions.py``.

Four triggers, each with a different shape of evidence:

``uncovered_topics``  requests the router could not place, clustered by the words they
                      share. One request nothing covered is a bad day; eight that use
                      the same vocabulary is a subject the system has no module for.
``saturated_module``  a module that has been trained repeatedly and whose held-out score
                      has stopped moving. More LoRA rank is the cheap thing to try.
``unused_module``     an adapter with no demand and no activations. Carrying it costs
                      VRAM in the warm cache and a slot in the adapter budget.
``merge_candidate``   a module that is always on, has been promoted repeatedly and is
                      stable. Folding it into the base frees its slot — and rewrites the
                      base checkpoint, which is why it can never happen unasked.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gustav.config.schema import ExpansionConfig
from gustav.inference.registry import Registry
from gustav.inference.router import tokenize
from gustav.store.db import Database

#: A word has to appear in at least this share of a cluster's requests to name it.
NAMING_SHARE = 0.5
#: Clusters are built around words that are not already a module's keyword.
MAX_CLUSTERS = 3
#: Question vocabulary, which is frequent in every cluster and names none of them.
#: Without this, "how do I write a kubernetes manifest" asked five ways produces a
#: module called "write".
GENERIC = frozenset(
    """question questions answer answers example examples thing things help way ways
    problem problems issue error errors work works working need want make making use
    using used get getting got best better good difference between please thanks
    number numbers sure know tell show explain understand possible able trying tried
    code file files line lines version versions time times new old same different
    """.split()
)


@dataclass
class Trigger:
    kind: str
    subject: str
    strength: float
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.kind}:{self.subject} ({self.strength:.2f})"


def _uncovered_requests(db: Database, *, limit: int = 500) -> list[str]:
    """The text of requests the router could not place above its threshold."""
    rows = db.query(
        "SELECT tags FROM telemetry WHERE metric = 'router.decision'"
        " ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    turn_ids: list[int] = []
    for row in rows:
        try:
            tags = json.loads(row["tags"] or "{}")
        except (ValueError, TypeError):
            continue
        if tags.get("uncovered") and tags.get("turn_id"):
            turn_ids.append(int(tags["turn_id"]))
    if not turn_ids:
        return []
    marks = ",".join("?" for _ in turn_ids)
    texts = db.query(
        f"SELECT content FROM turns WHERE id IN ({marks}) AND role = 'user'", turn_ids
    )
    return [row["content"] for row in texts]


def cluster_requests(
    requests: list[str], registry: Registry, *, min_share: float
) -> list[tuple[str, list[str], list[str]]]:
    """Group requests around the words they share.

    Returns ``(name, keywords, requests)`` per cluster. Words that are already some
    module's keyword are ignored for naming: the point is to find the subject nothing
    covers, and "python" is covered.
    """
    known = {k.lower() for module in registry for k in module.keywords}
    known |= {module.name.lower() for module in registry}
    counts: Counter[str] = Counter()
    per_request: list[set[str]] = []
    for request in requests:
        words = {w.strip(".-_") for w in tokenize(request)}
        words = {w for w in words if len(w) > 2 and w not in known and w not in GENERIC}
        per_request.append(words)
        counts.update(words)

    clusters: list[tuple[str, list[str], list[str]]] = []
    claimed: set[int] = set()
    # Deterministic order: most frequent, then longest, then alphabetical. Counter's
    # own most_common() breaks ties by insertion order, and the insertions come from
    # iterating sets — so the name of a cluster would otherwise depend on hash order.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    for word, count in ranked:
        if count < 2:
            break
        members = [
            index
            for index, words in enumerate(per_request)
            if word in words and index not in claimed
        ]
        if len(members) / max(len(requests), 1) < min_share and len(members) < 2:
            continue
        if not members:
            continue
        claimed.update(members)
        # The cluster's vocabulary: words shared by at least half of its requests.
        shared: Counter[str] = Counter()
        for index in members:
            shared.update(per_request[index])
        keywords = [
            w
            for w, n in shared.most_common(12)
            if n >= max(2, int(len(members) * NAMING_SHARE))
        ]
        clusters.append((word, keywords or [word], [requests[i] for i in members]))
        if len(clusters) >= MAX_CLUSTERS:
            break
    return clusters


def _gate_history(db: Database, module: str, *, limit: int = 6) -> list[float]:
    rows = db.query(
        "SELECT delta FROM gate_results WHERE module = ? AND suite = 'heldout'"
        " AND delta IS NOT NULL ORDER BY id DESC LIMIT ?",
        (module, limit),
    )
    return [float(r["delta"]) for r in rows]


def _demand(db: Database, module: str) -> float:
    row = db.query_one(
        "SELECT COALESCE(SUM(demand), 0) AS d, COALESCE(SUM(trials), 0) AS t"
        " FROM skills WHERE module = ?",
        (module,),
    )
    return float(row["d"] or 0.0) if row else 0.0


def _last_activation_days(db: Database, module: str) -> float | None:
    row = db.query_one(
        "SELECT (julianday('now') - julianday(MAX(ts))) AS days FROM telemetry"
        " WHERE metric = 'router.decision' AND tags LIKE ?",
        (f'%"{module}"%',),
    )
    if row is None or row["days"] is None:
        return None
    return float(row["days"])


def _promotions(db: Database, module: str) -> int:
    return int(
        db.scalar(
            "SELECT COUNT(*) FROM training_runs WHERE module = ? AND status = 'promoted'",
            (module,),
            default=0,
        )
        or 0
    )


def measure(
    db: Database, registry: Registry, config: ExpansionConfig, *, adapters: Any = None
) -> list[Trigger]:
    """Everything the store currently says about growing, strongest first."""
    found: list[Trigger] = []

    # --- a subject nothing covers --------------------------------------
    requests = _uncovered_requests(db)
    if len(requests) >= config.min_uncovered_requests:
        for name, keywords, members in cluster_requests(
            requests, registry, min_share=config.min_cluster_share
        ):
            if len(members) < max(2, config.min_uncovered_requests // 2):
                continue
            found.append(
                Trigger(
                    kind="uncovered_topics",
                    subject=name,
                    strength=min(len(members) / max(len(requests), 1), 1.0),
                    rationale=(
                        f"{len(members)} of the last {len(requests)} requests no module "
                        f"covered are about '{name}'"
                    ),
                    evidence={
                        "requests": len(members),
                        "uncovered_total": len(requests),
                        "keywords": keywords,
                        "examples": members[:3],
                    },
                )
            )

    # --- modules that stopped improving, are unused, or could be merged -
    for module in registry:
        history = _gate_history(db, module.name, limit=config.saturation_runs)
        demand = _demand(db, module.name)
        promotions = _promotions(db, module.name)

        if (
            len(history) >= config.saturation_runs
            and all(abs(delta) < config.saturation_delta for delta in history)
        ):
            found.append(
                Trigger(
                    kind="saturated_module",
                    subject=module.name,
                    strength=min(0.4 + demand, 1.0),
                    rationale=(
                        f"{module.name}'s last {len(history)} gate results moved the "
                        f"held-out score by less than {config.saturation_delta:+.2f}"
                    ),
                    evidence={"deltas": [round(d, 4) for d in history],
                              "rank": module.lora_rank, "demand": round(demand, 3)},
                )
            )

        versions = adapters.versions(module.name) if adapters is not None else []
        if versions and not module.always_on:
            idle_days = _last_activation_days(db, module.name)
            if demand <= 0.01 and (idle_days is None or idle_days >= config.unused_days):
                found.append(
                    Trigger(
                        kind="unused_module",
                        subject=module.name,
                        strength=0.3,
                        rationale=(
                            f"{module.name} has {len(versions)} adapter version(s), no "
                            f"measured demand and no routing hit in "
                            + (f"{idle_days:.0f} days" if idle_days else "the log")
                        ),
                        evidence={"versions": len(versions), "demand": round(demand, 3),
                                  "idle_days": idle_days},
                    )
                )

        if module.always_on and promotions >= 3 and versions:
            found.append(
                Trigger(
                    kind="merge_candidate",
                    subject=module.name,
                    strength=0.2 + min(promotions / 20, 0.5),
                    rationale=(
                        f"{module.name} is always on and has been promoted {promotions} "
                        f"times; folding it into the base would free its adapter slot"
                    ),
                    evidence={"promotions": promotions, "versions": len(versions)},
                )
            )

    found.sort(key=lambda t: t.strength, reverse=True)
    return found
