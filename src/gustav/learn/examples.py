"""The training-example store.

One table for everything the night produces — corrections, preference pairs, facts,
cited study answers, verified practice solutions — because the trainer's job is to mix
sources in known proportions, and it can only do that if they are comparable.

Three properties this file exists to guarantee:

* **Nothing is stored twice.** The fingerprint is a UNIQUE column, so rediscovering an
  example is a no-op rather than a quietly doubled weight.
* **The held-out split is stable.** Which side of the line an example falls on is
  derived from its fingerprint, not from a shuffle, so an example can never move into
  the training set on a later night and contaminate the gate that measured it.
* **Held-out examples are unreachable from the trainer.** ``for_training`` filters on
  ``split``; there is no code path that selects both.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from gustav.learn.fingerprint import bucket, fingerprint
from gustav.store.db import Database

#: Where an example came from, highest priority first. The weights are defaults that a
#: caller may override; the ordering is what the trainer uses to break ties.
KIND_WEIGHTS: dict[str, float] = {
    "correction": 10.0,   # the user told us the answer: nothing outranks this
    "preference": 2.0,    # /good, /bad
    "fact": 3.0,          # a durable statement about the user's world
    "practice": 2.0,      # solved a task and the verifier agreed
    "dream": 1.0,         # self-generated, verified by rejection sampling
    "study": 1.0,         # derived from documentation, with a citation
}

#: Kinds that are never held back from training. A correction is the user saying "this
#: is the answer"; holding it out to measure with would mean refusing to learn it.
ALWAYS_TRAIN = frozenset({"correction", "preference"})

_COLUMNS = (
    "id, created_at, run_id, module, kind, prompt, completion, rejected, weight,"
    " source, source_ref, citation, verified, split, status, trained_run_id, fingerprint"
)


@dataclass
class Example:
    prompt: str
    completion: str
    kind: str
    module: str = "general"
    rejected: str | None = None
    weight: float | None = None
    source: str | None = None
    source_ref: str | None = None
    citation: str | None = None
    verified: bool = False
    #: Leave as None to let the store assign it from the fingerprint.
    split: str | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.module, self.prompt, self.completion, self.rejected)

    def effective_weight(self) -> float:
        if self.weight is not None:
            return float(self.weight)
        return KIND_WEIGHTS.get(self.kind, 1.0)


@dataclass
class AddResult:
    added: list[int] = field(default_factory=list)
    duplicates: int = 0
    heldout: int = 0

    @property
    def count(self) -> int:
        return len(self.added)

    def merge(self, other: "AddResult") -> "AddResult":
        self.added.extend(other.added)
        self.duplicates += other.duplicates
        self.heldout += other.heldout
        return self


class ExampleStore:
    def __init__(self, db: Database, *, heldout_fraction: float = 0.15):
        self.db = db
        self.heldout_fraction = max(0.0, min(1.0, heldout_fraction))

    # --- writing --------------------------------------------------------
    def split_for(self, example: Example) -> str:
        if example.split is not None:
            return example.split
        if example.kind in ALWAYS_TRAIN or self.heldout_fraction <= 0:
            return "train"
        return "heldout" if bucket(example.fingerprint) < self.heldout_fraction else "train"

    def add(self, example: Example, *, run_id: int | None = None) -> int | None:
        """Store one example. Returns its id, or None if it was already known."""
        if not example.prompt.strip() or not example.completion.strip():
            return None
        split = self.split_for(example)
        try:
            return self.db.insert(
                "training_examples",
                {
                    "run_id": run_id,
                    "module": example.module,
                    "kind": example.kind,
                    "prompt": example.prompt,
                    "completion": example.completion,
                    "rejected": example.rejected,
                    "weight": example.effective_weight(),
                    "source": example.source,
                    "source_ref": example.source_ref,
                    "citation": example.citation,
                    "verified": int(bool(example.verified)),
                    "split": split,
                    "fingerprint": example.fingerprint,
                },
            )
        except Exception as exc:  # sqlite3.IntegrityError and friends
            if "UNIQUE" not in str(exc):
                raise
            return None

    def add_many(
        self, examples: Iterable[Example], *, run_id: int | None = None
    ) -> AddResult:
        result = AddResult()
        for example in examples:
            split = self.split_for(example)
            new_id = self.add(example, run_id=run_id)
            if new_id is None:
                result.duplicates += 1
                continue
            result.added.append(new_id)
            if split == "heldout":
                result.heldout += 1
        return result

    # --- reading --------------------------------------------------------
    def for_training(
        self,
        *,
        module: str | None = None,
        limit: int | None = None,
        verified_only: bool = False,
        include_used: bool = False,
    ) -> list[dict[str, Any]]:
        """Fresh training material, heaviest first. Never held-out, by construction."""
        sql = [f"SELECT {_COLUMNS} FROM training_examples WHERE split = 'train'"]
        params: list[Any] = []
        if include_used:
            sql.append("AND status IN ('ready', 'used')")
        else:
            sql.append("AND status = 'ready'")
        if module is not None:
            sql.append("AND module = ?")
            params.append(module)
        if verified_only:
            sql.append("AND verified = 1")
        sql.append("ORDER BY weight DESC, id ASC")
        if limit:
            sql.append("LIMIT ?")
            params.append(int(limit))
        return [dict(r) for r in self.db.query(" ".join(sql), params)]

    def replay(self, *, module: str | None = None, limit: int = 0) -> list[dict[str, Any]]:
        """Examples earlier nights already learned, for the anti-forgetting mix.

        Spread across what the module knows rather than taken newest-first, so replay
        keeps old skills alive instead of re-teaching last night's.
        """
        if limit <= 0:
            return []
        sql = [
            f"SELECT {_COLUMNS} FROM training_examples",
            "WHERE split = 'train' AND status = 'used'",
        ]
        params: list[Any] = []
        if module is not None:
            sql.append("AND module = ?")
            params.append(module)
        # Deterministic spread: order by the fingerprint, which is uncorrelated with
        # age, so the same call twice gives the same sample and it is not all recent.
        sql.append("ORDER BY fingerprint LIMIT ?")
        params.append(int(limit))
        return [dict(r) for r in self.db.query(" ".join(sql), params)]

    def heldout(
        self, *, module: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = [f"SELECT {_COLUMNS} FROM training_examples WHERE split = 'heldout'"]
        params: list[Any] = []
        if module is not None:
            sql.append("AND module = ?")
            params.append(module)
        sql.append("ORDER BY id DESC")
        if limit:
            sql.append("LIMIT ?")
            params.append(int(limit))
        return [dict(r) for r in self.db.query(" ".join(sql), params)]

    def modules_with_material(self, minimum: int = 1) -> dict[str, int]:
        rows = self.db.query(
            "SELECT module, COUNT(*) AS n FROM training_examples"
            " WHERE split = 'train' AND status = 'ready' GROUP BY module HAVING n >= ?",
            (int(minimum),),
        )
        return {row["module"]: int(row["n"]) for row in rows}

    def counts(self, *, run_id: int | None = None) -> dict[str, int]:
        if run_id is None:
            rows = self.db.query(
                "SELECT kind, COUNT(*) AS n FROM training_examples GROUP BY kind"
            )
        else:
            rows = self.db.query(
                "SELECT kind, COUNT(*) AS n FROM training_examples WHERE run_id = ?"
                " GROUP BY kind",
                (run_id,),
            )
        return {row["kind"]: int(row["n"]) for row in rows}

    # --- state ----------------------------------------------------------
    def mark_used(self, ids: Sequence[int], *, run_id: int | None = None) -> int:
        if not ids:
            return 0
        marks = ",".join("?" for _ in ids)
        cur = self.db.execute(
            f"UPDATE training_examples SET status = 'used', trained_run_id = ?"
            f" WHERE id IN ({marks}) AND status = 'ready'",
            [run_id, *ids],
        )
        return cur.rowcount or 0

    def drop_for_turns(self, turn_ids: Sequence[int], *, reason: str = "forgotten") -> int:
        """Withdraw examples derived from turns the user asked us to forget."""
        if not turn_ids:
            return 0
        refs = [str(int(t)) for t in turn_ids]
        marks = ",".join("?" for _ in refs)
        cur = self.db.execute(
            f"UPDATE training_examples SET status = 'dropped', citation = ?"
            f" WHERE source = 'turn' AND source_ref IN ({marks}) AND status = 'ready'",
            [reason, *refs],
        )
        return cur.rowcount or 0
