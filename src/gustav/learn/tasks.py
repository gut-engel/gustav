"""Practice tasks and attempts at them.

A task carries its own verifier, which is the whole point: "did it get this right" is
answered by running the tests or comparing against the answer, never by asking the
model what it thinks of its own work. Every attempt is recorded — including the failures
— because the failures are what `reflect` and the skill table are made of.

Verifier shapes, stored as JSON in ``tasks.verifier``:

``python``   ``{"setup": str, "tests": str, "entrypoint": str}`` — the answer's code is
             run with ``setup`` before and ``tests`` after it, in the sandbox.
``qa``       ``{"answer": str, "match": "contains" | "exact" | "number"}``
``rewrite``  ``{"rejected": str, "question": str}`` — an answer the user marked bad;
             a replacement has to be produced and agreed on before it is worth anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from gustav.learn.fingerprint import fingerprint
from gustav.store.db import Database

KINDS = ("python", "qa", "rewrite")


@dataclass
class Task:
    prompt: str
    kind: str
    verifier: dict[str, Any] = field(default_factory=dict)
    module: str = "general"
    topic: str | None = None
    difficulty: float = 0.5
    source: str | None = None
    source_ref: str | None = None
    document_id: int | None = None
    id: int | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.kind, self.module, self.prompt, json.dumps(
            self.verifier, sort_keys=True
        ))


def _to_task(row: Any) -> Task:
    try:
        verifier = json.loads(row["verifier"] or "{}")
    except (ValueError, TypeError):
        verifier = {}
    return Task(
        id=int(row["id"]),
        prompt=row["prompt"],
        kind=row["kind"],
        verifier=verifier if isinstance(verifier, dict) else {},
        module=row["module"],
        topic=row["topic"],
        difficulty=float(row["difficulty"]),
        source=row["source"],
        source_ref=row["source_ref"],
        document_id=row["document_id"],
    )


class TaskStore:
    def __init__(self, db: Database):
        self.db = db

    # --- writing --------------------------------------------------------
    def add(self, task: Task, *, run_id: int | None = None) -> int | None:
        if task.kind not in KINDS:
            raise ValueError(f"unknown task kind {task.kind!r}; expected one of {KINDS}")
        if not task.prompt.strip():
            return None
        try:
            return self.db.insert(
                "tasks",
                {
                    "run_id": run_id,
                    "module": task.module,
                    "topic": task.topic,
                    "kind": task.kind,
                    "prompt": task.prompt,
                    "verifier": json.dumps(task.verifier),
                    "difficulty": float(task.difficulty),
                    "source": task.source,
                    "source_ref": task.source_ref,
                    "document_id": task.document_id,
                    "fingerprint": task.fingerprint,
                },
            )
        except Exception as exc:
            if "UNIQUE" not in str(exc):
                raise
            return None

    def add_many(self, tasks: Iterable[Task], *, run_id: int | None = None) -> list[int]:
        return [new_id for t in tasks if (new_id := self.add(t, run_id=run_id)) is not None]

    # --- reading --------------------------------------------------------
    def open(
        self,
        *,
        kind: str | None = None,
        module: str | None = None,
        topics: Sequence[str] | None = None,
        limit: int = 50,
        run_id: int | None = None,
    ) -> list[Task]:
        sql = ["SELECT * FROM tasks WHERE status = 'open'"]
        params: list[Any] = []
        if kind:
            sql.append("AND kind = ?")
            params.append(kind)
        if module:
            sql.append("AND module = ?")
            params.append(module)
        if topics:
            marks = ",".join("?" for _ in topics)
            sql.append(f"AND topic IN ({marks})")
            params.extend(topics)
        if run_id is not None:
            sql.append("AND run_id = ?")
            params.append(run_id)
        # Hardest first: a task that has been attempted and failed is the one worth
        # the GPU, and a brand-new task (0 attempts) comes next.
        sql.append("ORDER BY (attempts > 0 AND passes = 0) DESC, difficulty DESC, id ASC LIMIT ?")
        params.append(int(limit))
        return [_to_task(r) for r in self.db.query(" ".join(sql), params)]

    def get(self, task_id: int) -> Task | None:
        row = self.db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return _to_task(row) if row else None

    def counts(self, *, run_id: int | None = None) -> dict[str, int]:
        if run_id is None:
            rows = self.db.query("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
        else:
            rows = self.db.query(
                "SELECT status, COUNT(*) AS n FROM tasks WHERE run_id = ? GROUP BY status",
                (run_id,),
            )
        return {r["status"]: int(r["n"]) for r in rows}

    # --- attempts -------------------------------------------------------
    def record_attempt(
        self,
        task_id: int,
        *,
        run_id: int | None,
        step: str,
        answer: str,
        passed: bool,
        detail: str = "",
        duration_ms: int | None = None,
    ) -> int:
        attempt_no = (
            int(
                self.db.scalar(
                    "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?", (task_id,), default=0
                )
                or 0
            )
            + 1
        )
        attempt_id = self.db.insert(
            "task_attempts",
            {
                "task_id": task_id,
                "run_id": run_id,
                "step": step,
                "attempt": attempt_no,
                "answer": answer,
                "passed": int(bool(passed)),
                "detail": detail[:2000],
                "duration_ms": duration_ms,
            },
        )
        self.db.execute(
            "UPDATE tasks SET attempts = attempts + 1, passes = passes + ?,"
            " status = CASE WHEN ? THEN 'solved' ELSE status END WHERE id = ?",
            (1 if passed else 0, 1 if passed else 0, task_id),
        )
        return attempt_id

    def set_status(self, task_id: int, status: str) -> None:
        self.db.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))

    def attempts(self, task_id: int) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query(
                "SELECT * FROM task_attempts WHERE task_id = ? ORDER BY id", (task_id,)
            )
        ]

    def results_for_run(self, run_id: int) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query(
                "SELECT a.*, t.module, t.topic, t.kind, t.prompt FROM task_attempts a"
                " JOIN tasks t ON t.id = a.task_id WHERE a.run_id = ? ORDER BY a.id",
                (run_id,),
            )
        ]
