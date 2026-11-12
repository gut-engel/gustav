"""The benchmark suites the gate measures against.

Two suites, with different jobs:

``heldout``     Material the trainer never saw, because the split was decided by the
                example's fingerprint before training began. It answers "did this
                adapter get better at the kind of thing we taught it".
``regression``  Things the model demonstrably *could* do before tonight: earlier
                corrections it was trained on, tasks it has passed. It answers "did
                tonight's training break something that worked". These items were
                trained on in earlier nights, deliberately — the question is retention,
                not generalisation, and a forgetting check has to use what was learned.

Scoring is per item kind, and never a model's opinion of its own answer:

``python``  the answer's code is run against the item's tests in the sandbox;
``qa``      exact, substring or numeric comparison with the expected answer.

For held-out *examples* (open-ended prose, no test) the score is token overlap with the
reference answer above a threshold. That is a weak metric and it is treated as one: it
is reported as a share, the gate needs a minimum number of items before it will believe
it at all, and anything close is a `skip` rather than a promotion.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from gustav.learn.fingerprint import fingerprint
from gustav.store.db import Database
from gustav.wake.implicit import similarity

#: Token-overlap above which a free-text answer counts as matching the reference.
PROSE_MATCH = 0.5


@dataclass
class BenchItem:
    prompt: str
    kind: str = "qa"
    expected: str = ""
    verifier: dict[str, Any] = field(default_factory=dict)
    module: str = "general"
    topic: str | None = None
    suite: str = "heldout"
    id: int | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.suite, self.module, self.prompt, self.expected)


def _to_item(row: Any) -> BenchItem:
    try:
        verifier = json.loads(row["verifier"] or "{}")
    except (ValueError, TypeError):
        verifier = {}
    return BenchItem(
        id=int(row["id"]),
        prompt=row["prompt"],
        kind=row["kind"],
        expected=row["expected"] or "",
        verifier=verifier if isinstance(verifier, dict) else {},
        module=row["module"],
        topic=row["topic"],
        suite=row["suite"],
    )


class BenchmarkStore:
    def __init__(self, db: Database):
        self.db = db

    def add(self, item: BenchItem, *, run_id: int | None = None) -> int | None:
        if not item.prompt.strip():
            return None
        try:
            return self.db.insert(
                "benchmark_items",
                {
                    "suite": item.suite,
                    "module": item.module,
                    "topic": item.topic,
                    "kind": item.kind,
                    "prompt": item.prompt,
                    "expected": item.expected,
                    "verifier": json.dumps(item.verifier),
                    "run_id": run_id,
                    "fingerprint": item.fingerprint,
                },
            )
        except Exception as exc:
            if "UNIQUE" not in str(exc):
                raise
            return None

    def add_many(self, items: Iterable[BenchItem], *, run_id: int | None = None) -> int:
        return sum(1 for i in items if self.add(i, run_id=run_id) is not None)

    def suite(
        self, suite: str, *, module: str | None = None, limit: int = 120
    ) -> list[BenchItem]:
        sql = ["SELECT * FROM benchmark_items WHERE suite = ? AND retired = 0"]
        params: list[Any] = [suite]
        if module:
            sql.append("AND module = ?")
            params.append(module)
        # Stable order: the same suite twice is the same measurement.
        sql.append("ORDER BY fingerprint LIMIT ?")
        params.append(int(limit))
        return [_to_item(r) for r in self.db.query(" ".join(sql), params)]

    def counts(self) -> dict[str, int]:
        rows = self.db.query(
            "SELECT suite, COUNT(*) AS n FROM benchmark_items WHERE retired = 0 GROUP BY suite"
        )
        return {r["suite"]: int(r["n"]) for r in rows}

    def retire(self, item_id: int) -> None:
        self.db.execute("UPDATE benchmark_items SET retired = 1 WHERE id = ?", (item_id,))


def score_answer(
    item: BenchItem,
    answer: str | None,
    *,
    sandbox: Any = None,
    matcher: Any = None,
) -> tuple[bool, str]:
    """Did this answer pass this item? Returns (passed, why).

    ``matcher`` is an optional semantic comparator for free-text answers. With one, a
    correct answer in different words passes; without one, the fallback is word overlap,
    which cannot tell that apart from a wrong answer.
    """
    if answer is None:
        return False, "no answer was produced"
    if item.kind == "python":
        from gustav.learn.code import extract_python

        code = extract_python(answer)
        if not code:
            return False, "no python that parses"
        if sandbox is None:
            return False, "no sandbox available to run the tests"
        result = sandbox.run_python(
            code,
            setup=str(item.verifier.get("setup") or ""),
            tests=str(item.verifier.get("tests") or ""),
        )
        return result.passed, result.detail
    expected = (item.expected or "").strip()
    if not expected:
        return False, "the item has no expected answer"
    mode = str(item.verifier.get("match") or "prose")
    given = answer.strip()
    if mode == "exact":
        passed = given.lower() == expected.lower()
    elif mode == "contains":
        passed = expected.lower() in given.lower()
    elif mode == "number":
        import re

        numbers = re.findall(r"-?\d+(?:\.\d+)?", given)
        wanted = re.findall(r"-?\d+(?:\.\d+)?", expected)
        passed = bool(wanted) and numbers[: len(wanted)] == wanted
    else:
        if matcher is not None:
            passed, score = matcher.matches(given, expected)
            return passed, f"semantic {score:.2f}"
        overlap = similarity(given, expected)
        return overlap >= PROSE_MATCH, f"overlap {overlap:.2f}"
    return passed, "passed" if passed else f"expected {expected[:80]!r}"


def seed_regression_suite(
    db: Database,
    *,
    module: str | None = None,
    limit: int = 60,
    run_id: int | None = None,
    exclude_run_id: int | None = None,
) -> int:
    """Promote already-learned material into the regression suite.

    Only examples with ``status = 'used'`` — trained on in an earlier night — and only
    the kinds where a right answer is actually known: a correction the user gave, a
    task solution a test accepted, a stated fact.

    ``exclude_run_id`` leaves out what *tonight* trained on. Measuring retention with
    material the candidate has just been trained on would flatter it, and the gate
    would then promote on a number it had been handed the answers to.
    """
    store = BenchmarkStore(db)
    sql = [
        "SELECT * FROM training_examples WHERE status = 'used' AND split = 'train'",
        "AND kind IN ('correction', 'practice', 'fact', 'dream')",
    ]
    params: list[Any] = []
    if exclude_run_id is not None:
        sql.append("AND (trained_run_id IS NULL OR trained_run_id != ?)")
        params.append(int(exclude_run_id))
    if module:
        sql.append("AND module = ?")
        params.append(module)
    sql.append("ORDER BY weight DESC, id ASC LIMIT ?")
    params.append(int(limit))
    added = 0
    for row in db.query(" ".join(sql), params):
        kind = "qa"
        verifier: dict[str, Any] = {"match": "prose"}
        if row["kind"] in ("practice", "dream") and row["source"] == "task":
            # source_ref is a text column, and SQLite will not match TEXT '5' against
            # INTEGER 5, so the id has to be converted rather than passed through.
            try:
                task_id = int(row["source_ref"])
            except (TypeError, ValueError):
                task_id = None
            task = (
                db.query_one("SELECT verifier FROM tasks WHERE id = ?", (task_id,))
                if task_id is not None
                else None
            )
        else:
            task = None
        if task is not None:
            try:
                spec = json.loads(task["verifier"] or "{}")
            except (ValueError, TypeError):
                spec = {}
            if spec.get("tests"):
                kind, verifier = "python", spec
        added += int(
            store.add(
                BenchItem(
                    prompt=row["prompt"],
                    kind=kind,
                    expected=row["completion"],
                    verifier=verifier,
                    module=row["module"],
                    suite="regression",
                ),
                run_id=run_id,
            )
            is not None
        )
    return added
