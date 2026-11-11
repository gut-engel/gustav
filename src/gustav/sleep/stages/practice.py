"""Practice: attempt tonight's tasks and let the verifier judge.

This step is a **measurement**, which is why it is greedy and single-attempt: one go at
each task at temperature 0, and the result goes into the skill table whether it passed
or not. If it sampled until something worked, the number it produced would describe the
sampler, not the model, and `reflect` would then choose tonight's topics from it.

Generating *data* by sampling until something passes is a different job with different
rules, and it is the next step (`dreaming`).

Every result is a trial against a (module, topic) pair, and a passing attempt also
becomes a training example — verified, because a test agreed, not because the model
sounded sure.
"""

from __future__ import annotations

from typing import Any

from gustav.learn.code import extract_python
from gustav.learn.examples import Example
from gustav.learn.tasks import Task, TaskStore
from gustav.sandbox import Sandbox, SandboxResult
from gustav.sleep.stages._base import StepEnv, sleep_step

CODE_SYSTEM = (
    "You are completing a small Python exercise. Reply with the implementation only, "
    "in one fenced python block, with no explanation. Define exactly the names the "
    "exercise asks for."
)
QA_SYSTEM = "Answer in one short sentence. No preamble."

MAX_TASKS = 40


def build_sandbox(env: StepEnv) -> Sandbox:
    cfg = env.config.sandbox
    return Sandbox(
        python=cfg.python,
        timeout_seconds=cfg.timeout_seconds,
        memory_limit_mb=cfg.memory_limit_mb,
        max_output_bytes=cfg.max_output_bytes,
        workdir=env.paths.tmp,
        isolate_env=cfg.isolate_env,
    )


def verify(task: Task, answer: str, sandbox: Sandbox) -> tuple[bool, str, str]:
    """(passed, what_was_submitted, detail). The only judge of a task's outcome."""
    if task.kind == "python":
        code = extract_python(answer)
        if not code:
            return False, answer, "the answer contained no python that parses"
        result: SandboxResult = sandbox.run_python(
            code,
            setup=str(task.verifier.get("setup") or ""),
            tests=str(task.verifier.get("tests") or ""),
        )
        return result.passed, code, result.detail
    if task.kind == "qa":
        expected = str(task.verifier.get("answer") or "").strip()
        mode = str(task.verifier.get("match") or "contains")
        given = (answer or "").strip()
        if not expected:
            return False, given, "the task has no expected answer"
        if mode == "exact":
            passed = given.lower() == expected.lower()
        elif mode == "number":
            passed = _numbers(given) == _numbers(expected) and bool(_numbers(expected))
        else:
            passed = expected.lower() in given.lower()
        return passed, given, "passed" if passed else f"expected {expected!r}"
    return False, answer, f"{task.kind} tasks are not judged by this step"


def _numbers(text: str) -> list[str]:
    import re

    return re.findall(r"-?\d+(?:\.\d+)?", text or "")


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    tasks = TaskStore(env.db)
    model = env.model()
    topics_payload = env.read_step_json("reflect", "topics.json", {}) or {}
    topics = [t["topic"] for t in topics_payload.get("topics", [])]

    result: dict[str, Any] = {
        "attempted": 0,
        "passed": 0,
        "failed": 0,
        "examples": 0,
        "model": model.available,
        "by_topic": {},
    }
    if not model.available:
        result["reason"] = model.error or "no model available"
        env.log(f"skipping practice: {result['reason']}", level="warning")
        return result

    # Tonight's topics first, then anything else open, so a night with no topics still
    # practises rather than idling. Asked per kind, because a pile of rewrite tasks —
    # which belong to `dreaming`, not here — would otherwise eat the whole budget.
    pool: list[Task] = []
    seen: set[int | None] = set()
    for kind in ("python", "qa"):
        for topic_filter in (topics or None, None):
            for task in tasks.open(
                kind=kind, topics=topic_filter, limit=MAX_TASKS - len(pool)
            ):
                if task.id not in seen:
                    seen.add(task.id)
                    pool.append(task)
            if len(pool) >= MAX_TASKS:
                break

    if not pool:
        result["reason"] = "no open tasks"
        return result
    if env.dry_run:
        result["dry_run"] = True
        result["attempted"] = len(pool)
        return result

    sandbox = build_sandbox(env)
    for _index, task in env.each(pool, label="task", checkpoint_key="task_index"):
        system = CODE_SYSTEM if task.kind == "python" else QA_SYSTEM
        answer = model.ask(task.prompt, system=system, temperature=0.0)
        if answer is None:
            result["failed"] += 1
            continue
        passed, submitted, detail = verify(task, answer, sandbox)
        result["attempted"] += 1
        result["passed" if passed else "failed"] += 1
        tasks.record_attempt(
            task.id or 0,
            run_id=env.run_id,
            step=env.step,
            answer=submitted,
            passed=passed,
            detail=detail,
        )
        if task.topic:
            env.skills.record(task.module, task.topic, passed=passed)
            bucket = result["by_topic"].setdefault(
                f"{task.module}/{task.topic}", {"passed": 0, "failed": 0}
            )
            bucket["passed" if passed else "failed"] += 1
        if passed:
            added = env.examples.add(
                Example(
                    prompt=task.prompt,
                    completion=submitted,
                    kind="practice",
                    module=task.module,
                    source="task",
                    source_ref=str(task.id),
                    verified=True,  # a test agreed, not the model
                ),
                run_id=env.run_id,
            )
            result["examples"] += 1 if added else 0

    env.write_json("practice.json", result)
    env.events.emit(
        "sleep.practised",
        f"{result['passed']}/{result['attempted']} tasks passed",
        run_id=env.run_id,
        passed=result["passed"],
        attempted=result["attempted"],
    )
    return result
