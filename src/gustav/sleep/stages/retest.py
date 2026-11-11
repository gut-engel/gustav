"""Retest: ask the things it got wrong again, with tonight's weights.

The gate measures a candidate on suites. This step asks a narrower and more useful
question: *the specific things the user corrected, and the specific tasks that failed —
can it do them now?* The answer goes into the skill table, which is what `reflect`
reads next time, so a topic that was fixed stops being studied and a topic that was
not keeps its place in the queue.

It runs after promotion on purpose. These trials are meant to describe the weights that
are actually being served from now on, not a candidate that may have been rejected.
"""

from __future__ import annotations

from typing import Any

from gustav.inference.registry import load_registry
from gustav.learn.bench import PROSE_MATCH
from gustav.learn.tasks import TaskStore
from gustav.learn.topics import detect as detect_topics
from gustav.sleep.stages._base import StepEnv, sleep_step
from gustav.sleep.stages.practice import CODE_SYSTEM, build_sandbox, verify
from gustav.wake.implicit import similarity

MAX_ITEMS = 30


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    model = env.model()
    tasks = TaskStore(env.db)
    registry = load_registry(env.paths.registry)
    result: dict[str, Any] = {
        "corrections_retested": 0,
        "corrections_now_right": 0,
        "tasks_retested": 0,
        "tasks_now_passing": 0,
        "model": model.available,
        "topics": {},
    }
    if not model.available:
        result["reason"] = model.error or "no model available"
        return result

    corrections = env.db.query(
        "SELECT prompt, completion, module FROM training_examples"
        " WHERE run_id = ? AND kind = 'correction' LIMIT ?",
        (env.run_id, MAX_ITEMS),
    )
    failures = env.db.query(
        "SELECT DISTINCT t.id FROM task_attempts a JOIN tasks t ON t.id = a.task_id"
        " WHERE a.run_id = ? AND a.passed = 0 AND t.kind = 'python' LIMIT ?",
        (env.run_id, MAX_ITEMS),
    )
    if not corrections and not failures:
        result["reason"] = "nothing failed tonight that is worth asking again"
        return result
    if env.dry_run:
        result["dry_run"] = True
        result["corrections_retested"] = len(corrections)
        result["tasks_retested"] = len(failures)
        return result

    # --- the answers the user had to correct ----------------------------
    if corrections:
        answers = model.evaluate(
            [row["prompt"] for row in corrections], temperature=0.0, max_tokens=256
        )
        for row, answer in zip(corrections, answers, strict=True):
            result["corrections_retested"] += 1
            agreement = similarity(answer or "", row["completion"])
            right = agreement >= PROSE_MATCH
            result["corrections_now_right"] += int(right)
            for topic in detect_topics(row["prompt"], registry, limit=1):
                env.skills.record(topic.module, topic.topic, passed=right)
                bucket = result["topics"].setdefault(str(topic), {"passed": 0, "failed": 0})
                bucket["passed" if right else "failed"] += 1

    # --- the tasks that failed ------------------------------------------
    if failures:
        sandbox = build_sandbox(env)
        for _index, row in env.each(list(failures), label="task", checkpoint_key="task_index"):
            task = tasks.get(int(row["id"]))
            if task is None:
                continue
            answer = model.ask(task.prompt, system=CODE_SYSTEM, temperature=0.0)
            passed, submitted, detail = verify(task, answer or "", sandbox)
            result["tasks_retested"] += 1
            result["tasks_now_passing"] += int(passed)
            tasks.record_attempt(
                task.id or 0, run_id=env.run_id, step=env.step, answer=submitted,
                passed=passed, detail=detail,
            )
            if task.topic:
                env.skills.record(task.module, task.topic, passed=passed)
                bucket = result["topics"].setdefault(
                    f"{task.module}/{task.topic}", {"passed": 0, "failed": 0}
                )
                bucket["passed" if passed else "failed"] += 1

    for label in result["topics"]:
        module, _, topic = label.partition("/")
        env.skills.mark_studied(module, topic)

    result["skills"] = env.skills.summary()
    env.write_json("retest.json", result)
    env.events.emit(
        "sleep.retested",
        f"{result['corrections_now_right']}/{result['corrections_retested']} corrected "
        f"answers right now, {result['tasks_now_passing']}/{result['tasks_retested']} "
        f"failed tasks passing",
        run_id=env.run_id,
        **{k: result[k] for k in ("corrections_now_right", "tasks_now_passing")},
    )
    return result
