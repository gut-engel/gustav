"""The measured skill table.

This is what turns "it seems worse at Rust lifetimes" into a number. Every verified
attempt — a practice task, a gate item, a retested failure — lands here as a trial
against a (module, topic) pair, and `reflect` ranks topics to study on it.

The score is Laplace-smoothed on purpose: one lucky pass out of one trial is not a
skill, and the smoothing keeps a single result from promoting a topic out of the
study list before it has been measured properly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from gustav.store.db import Database


@dataclass
class Skill:
    module: str
    topic: str
    trials: int
    passes: int
    score: float
    demand: float
    studied_at: str | None = None

    @property
    def weakness(self) -> float:
        """What `reflect` sorts on: how badly it does something it is actually asked.

        Demand matters as much as the score. Being poor at a topic nobody raises is
        not worth a night of training, and the whole point of the table is to spend
        the GPU on what the user actually needs.
        """
        return (1.0 - self.score) * (0.25 + self.demand)


def smoothed(passes: int, trials: int) -> float:
    """Laplace-smoothed pass rate. Unmeasured is 0.5, not 0."""
    return (passes + 1.0) / (trials + 2.0)


class SkillStore:
    def __init__(self, db: Database):
        self.db = db

    def _ensure(self, module: str, topic: str) -> None:
        self.db.execute(
            "INSERT INTO skills (module, topic, score) VALUES (?, ?, 0.5)"
            " ON CONFLICT(module, topic) DO NOTHING",
            (module, topic),
        )

    def record(self, module: str, topic: str, *, passed: bool, trials: int = 1) -> None:
        """One or more verified attempts at a topic."""
        self._ensure(module, topic)
        self.db.execute(
            "UPDATE skills SET trials = trials + ?, passes = passes + ?,"
            " last_seen_at = datetime('now') WHERE module = ? AND topic = ?",
            (int(trials), int(trials) if passed else 0, module, topic),
        )
        self._rescore(module, topic)

    def record_many(self, results: Iterable[tuple[str, str, bool]]) -> None:
        for module, topic, passed in results:
            self.record(module, topic, passed=passed)

    def _rescore(self, module: str, topic: str) -> None:
        row = self.db.query_one(
            "SELECT trials, passes FROM skills WHERE module = ? AND topic = ?", (module, topic)
        )
        if row is None:
            return
        self.db.execute(
            "UPDATE skills SET score = ? WHERE module = ? AND topic = ?",
            (smoothed(int(row["passes"]), int(row["trials"])), module, topic),
        )

    def note_demand(self, module: str, topic: str, *, hits: int = 1) -> None:
        """The user asked about this. Demand decays, so old interests fade out."""
        self._ensure(module, topic)
        self.db.execute(
            "UPDATE skills SET demand = MIN(1.0, demand * 0.98 + ?),"
            " last_seen_at = datetime('now') WHERE module = ? AND topic = ?",
            (0.1 * hits, module, topic),
        )

    def mark_studied(self, module: str, topic: str) -> None:
        self._ensure(module, topic)
        self.db.execute(
            "UPDATE skills SET studied_at = datetime('now') WHERE module = ? AND topic = ?",
            (module, topic),
        )

    def all(self) -> list[Skill]:
        return [self._to_skill(r) for r in self.db.query("SELECT * FROM skills ORDER BY module, topic")]

    def get(self, module: str, topic: str) -> Skill | None:
        row = self.db.query_one(
            "SELECT * FROM skills WHERE module = ? AND topic = ?", (module, topic)
        )
        return self._to_skill(row) if row else None

    def weakest(self, *, limit: int = 5, threshold: float = 0.6) -> list[Skill]:
        """Topics *measured* below the threshold, ranked by weakness.

        Unmeasured topics are excluded even though their smoothed score sits below a
        typical threshold. "We have never tested this" and "it fails at this" are
        different problems with different answers, and `reflect` handles them
        separately — conflating them would fill every night's study list with topics
        nobody has ever checked.
        """
        skills = [s for s in self.all() if s.trials > 0 and s.score < threshold]
        skills.sort(key=lambda s: s.weakness, reverse=True)
        return skills[:limit]

    def summary(self) -> dict[str, Any]:
        skills = self.all()
        measured = [s for s in skills if s.trials > 0]
        return {
            "topics": len(skills),
            "measured": len(measured),
            "mean_score": (
                round(sum(s.score for s in measured) / len(measured), 3) if measured else None
            ),
            "weakest": [
                {"module": s.module, "topic": s.topic, "score": round(s.score, 3),
                 "trials": s.trials}
                for s in sorted(measured, key=lambda s: s.score)[:5]
            ],
        }

    @staticmethod
    def _to_skill(row: Any) -> Skill:
        return Skill(
            module=row["module"],
            topic=row["topic"],
            trials=int(row["trials"]),
            passes=int(row["passes"]),
            score=float(row["score"]),
            demand=float(row["demand"]),
            studied_at=row["studied_at"],
        )
