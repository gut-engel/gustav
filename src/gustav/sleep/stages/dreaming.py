"""Dreaming: sample until something verifiably works, and keep only that.

The same tasks `practice` just measured, but with the opposite purpose: this step is
trying to *produce data*, so it samples several times at a temperature and keeps what
survives. That is rejection sampling, and the rejection is the valuable part — the
discarded attempts are the reason the kept ones are worth training on.

Two kinds of material come out of it:

* **Solved exercises.** A task `practice` failed at temperature 0 is attempted a few
  more times with sampling; a sample that passes the tests becomes a verified training
  example. Learning from a success it could only reach by luck is how it stops needing
  the luck.
* **Replacements for answers the user rejected.** A `/bad` gives us a question and an
  answer that was wrong, and no right answer. So several candidates are sampled and
  compared with each other: if they agree with one another and disagree with the
  rejected answer, the one most representative of that agreement becomes the preferred
  side of a preference pair. Self-consistency is *evidence*, not proof, so these are
  stored unverified and carry no more weight than a thumbs-up.

Nothing here is trusted because the model produced it. Every python example in this
step passed a test, and every preference pair is marked as what it is.
"""

from __future__ import annotations

from typing import Any

from gustav.learn.code import extract_python
from gustav.learn.examples import Example
from gustav.learn.tasks import Task, TaskStore
from gustav.sleep.stages._base import StepEnv, sleep_step
from gustav.sleep.stages.practice import CODE_SYSTEM, build_sandbox, verify
from gustav.wake.implicit import similarity

#: Samples per task. Small on purpose: four tries at 0.8 finds what one try at 0 missed
#: often enough to be worth it, and a night is not an afternoon of compute.
SAMPLES = 4
TEMPERATURE = 0.8
MAX_TASKS = 20

#: A set of candidate answers has to agree with itself at least this much before any of
#: them is treated as evidence of the right answer.
MIN_AGREEMENT = 0.45
#: ... and the winner must not simply restate the answer the user rejected.
MAX_SIMILARITY_TO_REJECTED = 0.8

REWRITE_SYSTEM = (
    "Answer the user's question directly and correctly. Be brief and concrete. "
    "Do not apologise and do not mention that an earlier answer was wrong."
)


def medoid(candidates: list[str]) -> tuple[str, float]:
    """The candidate most like all the others, and how much they agree.

    With one candidate there is no agreement to measure, so it scores 0 and will be
    rejected by the caller's threshold — which is the honest answer.
    """
    if not candidates:
        return "", 0.0
    if len(candidates) == 1:
        return candidates[0], 0.0
    best, best_score = candidates[0], -1.0
    for index, candidate in enumerate(candidates):
        # By position, not identity: four samples that came back word for word
        # identical are the strongest agreement there is, not an empty comparison.
        others = [c for i, c in enumerate(candidates) if i != index]
        score = sum(similarity(candidate, other) for other in others) / len(others)
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


def _solve_by_sampling(
    env: StepEnv, model: Any, sandbox: Any, task: Task, tasks: TaskStore, result: dict[str, Any]
) -> bool:
    for sample in range(SAMPLES):
        env.raise_if_stopping()
        answer = model.ask(
            task.prompt,
            system=CODE_SYSTEM,
            temperature=TEMPERATURE,
            top_p=0.95,
            seed=1000 + sample,
        )
        if not answer:
            continue
        result["samples"] += 1
        passed, submitted, detail = verify(task, answer, sandbox)
        tasks.record_attempt(
            task.id or 0,
            run_id=env.run_id,
            step=env.step,
            answer=submitted,
            passed=passed,
            detail=detail,
        )
        if not passed:
            result["rejected"] += 1
            continue
        added = env.examples.add(
            Example(
                prompt=task.prompt,
                completion=submitted,
                kind="dream",
                module=task.module,
                source="task",
                source_ref=str(task.id),
                verified=True,  # it passed the task's own tests
            ),
            run_id=env.run_id,
        )
        result["solved"] += 1
        result["examples"] += 1 if added else 0
        return True
    return False


def _rewrite(env: StepEnv, model: Any, task: Task, tasks: TaskStore, result: dict[str, Any]) -> bool:
    question = str(task.verifier.get("question") or task.prompt)
    rejected = str(task.verifier.get("rejected") or "")
    candidates: list[str] = []
    for sample in range(SAMPLES):
        env.raise_if_stopping()
        answer = model.ask(
            question, system=REWRITE_SYSTEM, temperature=TEMPERATURE, seed=2000 + sample
        )
        if answer and len(answer) > 8:
            candidates.append(answer)
            result["samples"] += 1

    winner, agreement = medoid(candidates)
    too_similar = bool(rejected) and similarity(winner, rejected) > MAX_SIMILARITY_TO_REJECTED
    if not winner or agreement < MIN_AGREEMENT or too_similar:
        result["rejected"] += 1
        tasks.record_attempt(
            task.id or 0,
            run_id=env.run_id,
            step=env.step,
            answer=winner,
            passed=False,
            detail=(
                f"candidates agreed {agreement:.2f}"
                + (", and the best one restates the rejected answer" if too_similar else "")
            ),
        )
        return False

    added = env.examples.add(
        Example(
            prompt=question,
            completion=winner,
            kind="preference",
            module=task.module,
            rejected=rejected or None,
            weight=2.0,
            source="task",
            source_ref=str(task.id),
            # Agreement between samples is evidence, not a verdict. Only a test or the
            # user sets this flag.
            verified=False,
        ),
        run_id=env.run_id,
    )
    tasks.record_attempt(
        task.id or 0, run_id=env.run_id, step=env.step, answer=winner, passed=True,
        detail=f"candidates agreed {agreement:.2f}",
    )
    tasks.set_status(task.id or 0, "solved")
    result["rewritten"] += 1
    result["examples"] += 1 if added else 0
    return True


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    tasks = TaskStore(env.db)
    model = env.model()
    result: dict[str, Any] = {
        "samples": 0,
        "solved": 0,
        "rewritten": 0,
        "rejected": 0,
        "examples": 0,
        "model": model.available,
    }
    if not model.available:
        result["reason"] = model.error or "no model available"
        return result

    # Tasks worth sampling at: the ones that have been tried and are still unsolved.
    hard = [t for t in tasks.open(kind="python", limit=MAX_TASKS)]
    rewrites = tasks.open(kind="rewrite", limit=MAX_TASKS)
    pool: list[Task] = hard + list(rewrites)
    if not pool:
        result["reason"] = "nothing left to dream about"
        return result
    if env.dry_run:
        result["dry_run"] = True
        result["pool"] = len(pool)
        return result

    sandbox = build_sandbox(env)
    for _index, task in env.each(pool, label="task", checkpoint_key="task_index"):
        if task.kind == "rewrite":
            _rewrite(env, model, task, tasks, result)
        else:
            _solve_by_sampling(env, model, sandbox, task, tasks, result)

    env.write_json("dreaming.json", result)
    env.events.emit(
        "sleep.dreamt",
        f"{result['solved']} tasks solved by sampling, {result['rewritten']} answers "
        f"replaced, {result['rejected']} candidates thrown away",
        run_id=env.run_id,
        **{k: result[k] for k in ("solved", "rewritten", "rejected", "samples")},
    )
    return result
