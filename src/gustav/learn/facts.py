"""Durable statements about the user's world.

A fact is never edited in place. A newer statement about the same subject *supersedes*
the old one, which leaves the history readable — "it used to think my editor was vim"
is answerable — and makes a wrong extraction retractable rather than invisible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from gustav.learn.fingerprint import fingerprint
from gustav.store.db import Database


@dataclass
class Fact:
    text: str
    subject: str | None = None
    module: str = "personal-memory"
    confidence: float = 0.6
    source: str | None = None
    source_turn_id: int | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.module, self.subject, self.text)


@dataclass
class FactChange:
    fact_id: int
    superseded: list[int]
    is_new: bool


class FactStore:
    def __init__(self, db: Database):
        self.db = db

    def add(
        self, fact: Fact, *, run_id: int | None = None, supersede: bool = True
    ) -> FactChange | None:
        """Store a fact. Returns None if this exact statement is already active."""
        text = fact.text.strip()
        if not text:
            return None
        existing = self.db.query_one(
            "SELECT id, status FROM facts WHERE fingerprint = ?", (fact.fingerprint,)
        )
        if existing is not None and existing["status"] == "active":
            # Seeing it again is weak evidence that it is true.
            self.db.execute(
                "UPDATE facts SET confidence = MIN(0.99, confidence + 0.05),"
                " updated_at = datetime('now') WHERE id = ?",
                (existing["id"],),
            )
            return None

        # Anything else active about the same subject is now out of date — including
        # when the statement being restored is one we superseded earlier. "My editor is
        # vim, actually no, helix again" has to end with one active editor, not two.
        superseded: list[int] = []
        if supersede and fact.subject:
            rows = self.db.query(
                "SELECT id FROM facts WHERE status = 'active' AND module = ? AND subject = ?"
                " AND id IS NOT ?",
                (fact.module, fact.subject, existing["id"] if existing else None),
            )
            superseded = [int(r["id"]) for r in rows]

        if existing is not None:
            fact_id = int(existing["id"])
            self.db.execute(
                "UPDATE facts SET status = 'active', updated_at = datetime('now'),"
                " superseded_by = NULL, confidence = ? WHERE id = ?",
                (float(fact.confidence), fact_id),
            )
            is_new = False
        else:
            fact_id = self.db.insert(
                "facts",
                {
                    "module": fact.module,
                    "subject": fact.subject,
                    "text": text,
                    "confidence": float(fact.confidence),
                    "source": fact.source,
                    "source_turn_id": fact.source_turn_id,
                    "run_id": run_id,
                    "fingerprint": fact.fingerprint,
                },
            )
            is_new = True

        for old_id in superseded:
            self.db.execute(
                "UPDATE facts SET status = 'superseded', superseded_by = ?,"
                " updated_at = datetime('now') WHERE id = ?",
                (fact_id, old_id),
            )
        return FactChange(fact_id, superseded, is_new=is_new)

    def add_many(self, facts: Iterable[Fact], *, run_id: int | None = None) -> list[FactChange]:
        return [c for f in facts if (c := self.add(f, run_id=run_id)) is not None]

    def active(self, *, module: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM facts WHERE status = 'active'"
        params: list[Any] = []
        if module:
            sql += " AND module = ?"
            params.append(module)
        sql += " ORDER BY confidence DESC, id DESC LIMIT ?"
        params.append(int(limit))
        return [dict(r) for r in self.db.query(sql, params)]

    def history(self, subject: str, *, module: str = "personal-memory") -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query(
                "SELECT * FROM facts WHERE module = ? AND subject = ? ORDER BY id",
                (module, subject),
            )
        ]

    def retract(self, fact_id: int, *, note: str = "") -> bool:
        cur = self.db.execute(
            "UPDATE facts SET status = 'retracted', updated_at = datetime('now')"
            " WHERE id = ? AND status != 'retracted'",
            (fact_id,),
        )
        if note:
            self.db.execute("UPDATE facts SET source = ? WHERE id = ?", (note, fact_id))
        return bool(cur.rowcount)

    def retract_for_turns(self, turn_ids: Iterable[int]) -> int:
        ids = [int(t) for t in turn_ids]
        if not ids:
            return 0
        marks = ",".join("?" for _ in ids)
        cur = self.db.execute(
            f"UPDATE facts SET status = 'retracted', updated_at = datetime('now')"
            f" WHERE source_turn_id IN ({marks}) AND status = 'active'",
            ids,
        )
        return cur.rowcount or 0

    def count(self, status: str = "active") -> int:
        return int(
            self.db.scalar("SELECT COUNT(*) FROM facts WHERE status = ?", (status,), default=0)
        )
