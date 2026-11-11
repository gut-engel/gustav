"""What happened while it was awake, as the night sees it.

The wake phase writes turns and feedback; consolidation reads them. The pairing is
done here, once, because every later step wants the same thing: a question, the answer
that was given, and whether we are allowed to learn from it.

``trainable = 0`` is respected at the join, not filtered later. A turn marked
never-train — a detected secret, a `/forget` — must not be able to reach a dataset
through a path that forgot to check, so an exchange with either side untrainable is not
an exchange at all as far as this module is concerned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gustav.store.db import Database

META_CONSOLIDATED_THROUGH = "sleep.consolidated_through_turn"


@dataclass
class Exchange:
    session_id: int
    question_turn_id: int
    question: str
    answer_turn_id: int | None = None
    answer: str = ""
    ts: str = ""
    adapters: dict[str, str] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return bool(self.answer.strip()) and self.answer_turn_id is not None


def exchanges(
    db: Database,
    *,
    after_turn_id: int = 0,
    since: str | None = None,
    limit: int = 2000,
) -> list[Exchange]:
    """Question/answer pairs we may learn from, oldest first."""
    sql = [
        "SELECT id, session_id, role, content, ts, trainable, adapter_versions",
        "FROM turns WHERE id > ?",
    ]
    params: list[Any] = [int(after_turn_id)]
    if since:
        sql.append("AND ts >= ?")
        params.append(since)
    sql.append("ORDER BY session_id, id LIMIT ?")
    params.append(int(limit) * 2)
    rows = db.query(" ".join(sql), params)

    found: list[Exchange] = []
    pending: dict[int, Any] = {}
    for row in rows:
        session = int(row["session_id"])
        if row["role"] == "user":
            pending[session] = row if row["trainable"] else None
            continue
        if row["role"] != "assistant":
            continue
        question = pending.pop(session, None)
        if question is None or not row["trainable"]:
            continue
        import json as _json

        try:
            adapters = _json.loads(row["adapter_versions"] or "{}")
        except (ValueError, TypeError):
            adapters = {}
        found.append(
            Exchange(
                session_id=session,
                question_turn_id=int(question["id"]),
                question=question["content"],
                answer_turn_id=int(row["id"]),
                answer=row["content"],
                ts=row["ts"],
                adapters=adapters if isinstance(adapters, dict) else {},
            )
        )
    found.sort(key=lambda e: e.question_turn_id)
    return found[:limit]


def unanswered_questions(db: Database, *, after_turn_id: int = 0, limit: int = 200) -> list[str]:
    """User messages that never got an answer — a crash, a preemption, a /forget."""
    rows = db.query(
        "SELECT t.content FROM turns t WHERE t.role = 'user' AND t.trainable = 1"
        " AND t.id > ? AND NOT EXISTS ("
        "   SELECT 1 FROM turns a WHERE a.session_id = t.session_id AND a.id > t.id"
        "   AND a.role = 'assistant')"
        " ORDER BY t.id LIMIT ?",
        (int(after_turn_id), int(limit)),
    )
    return [r["content"] for r in rows]


def last_turn_id(db: Database) -> int:
    return int(db.scalar("SELECT COALESCE(MAX(id), 0) FROM turns", default=0) or 0)


def consolidated_through(db: Database) -> int:
    return int(db.get_meta(META_CONSOLIDATED_THROUGH, 0) or 0)


def set_consolidated_through(db: Database, turn_id: int) -> None:
    db.set_meta(META_CONSOLIDATED_THROUGH, int(turn_id))


def open_feedback(db: Database, *, limit: int = 500) -> list[dict[str, Any]]:
    """Feedback no run has consumed yet, with the turn it refers to attached.

    ``consumed_run_id = -1`` is how `/forget` withdraws feedback, so it is excluded by
    the same condition that excludes feedback an earlier night already used.
    """
    rows = db.query(
        "SELECT f.id, f.kind, f.payload, f.source, f.turn_id, f.ts,"
        "       t.content AS answer, t.session_id, t.trainable,"
        "       (SELECT content FROM turns q WHERE q.session_id = t.session_id"
        "          AND q.id < t.id AND q.role = 'user' AND q.trainable = 1"
        "        ORDER BY q.id DESC LIMIT 1) AS question"
        " FROM feedback f LEFT JOIN turns t ON t.id = f.turn_id"
        " WHERE f.consumed_run_id IS NULL ORDER BY f.id LIMIT ?",
        (int(limit),),
    )
    return [dict(r) for r in rows]


def mark_feedback_consumed(db: Database, ids: list[int], run_id: int) -> int:
    if not ids:
        return 0
    marks = ",".join("?" for _ in ids)
    cur = db.execute(
        f"UPDATE feedback SET consumed_run_id = ?, reviewed = 1 WHERE id IN ({marks})",
        [run_id, *ids],
    )
    return cur.rowcount or 0
