"""Reflect: name the topics tonight is about.

Everything downstream — which documentation is fetched, which tasks are practised,
which modules are trained — follows from this list, so it is derived from measurements
rather than from the model's opinion of its own weaknesses:

* a `/correct` is a **failed trial** at the topic of the question, and a `/good` is a
  passed one: the user is a verifier, and the cheapest one available;
* a practice task that failed is a failed trial, which is where the score comes from on
  nights when the user said nothing;
* a topic that is *in demand but never measured* is a candidate too — not knowing
  whether we are good at something the user keeps asking about is itself a weakness;
* a request no module covered is recorded as a **gap**. Gaps are not studied here;
  they are the measured trigger for self-expansion in stage 6, and they go in the
  report so the decision stays the user's.

The ranking is ``(1 - score) * (0.25 + demand)``: being bad at something nobody asks
about does not earn a night of GPU time.
"""

from __future__ import annotations

import json
from typing import Any

from gustav.inference.registry import load_registry
from gustav.learn import day
from gustav.learn.topics import detect as detect_topics
from gustav.sleep.stages._base import StepEnv, sleep_step

#: Examples of this kind mean the model got it wrong / right, as judged by the user.
FAILED_KINDS = ("correction",)
PASSED_KINDS = ("preference",)


def _record_user_verdicts(env: StepEnv, registry: Any) -> dict[str, int]:
    rows = env.db.query(
        "SELECT prompt, kind FROM training_examples WHERE run_id = ? AND kind IN"
        " ('correction', 'preference')",
        (env.run_id,),
    )
    recorded = {"failed": 0, "passed": 0}
    for row in rows:
        passed = row["kind"] in PASSED_KINDS
        for topic in detect_topics(row["prompt"], registry, limit=2):
            env.skills.record(topic.module, topic.topic, passed=passed)
            recorded["passed" if passed else "failed"] += 1
    return recorded


def _record_task_results(env: StepEnv) -> int:
    """Attempts from the previous night that no one has scored yet."""
    rows = env.db.query(
        "SELECT a.id, a.passed, t.module, t.topic FROM task_attempts a"
        " JOIN tasks t ON t.id = a.task_id"
        " WHERE a.run_id < ? AND a.run_id = ("
        "   SELECT MAX(run_id) FROM task_attempts WHERE run_id < ?)",
        (env.run_id, env.run_id),
    )
    for row in rows:
        if row["topic"]:
            env.skills.record(row["module"], row["topic"], passed=bool(row["passed"]))
    return len(rows)


def _gaps(env: StepEnv, limit: int = 20) -> list[dict[str, Any]]:
    """Requests the router could not place. The trigger for self-expansion."""
    rows = env.db.query(
        "SELECT value, tags FROM telemetry WHERE metric = 'router.decision'"
        " ORDER BY id DESC LIMIT 500"
    )
    found: list[dict[str, Any]] = []
    for row in rows:
        try:
            tags = json.loads(row["tags"] or "{}")
        except (ValueError, TypeError):
            continue
        if tags.get("uncovered"):
            found.append({"confidence": round(float(row["value"]), 3),
                          "turn_id": tags.get("turn_id")})
    return found[:limit]


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    cfg = env.config.sleep
    registry = load_registry(env.paths.registry)

    verdicts = _record_user_verdicts(env, registry)
    scored_attempts = _record_task_results(env)
    env.progress(
        f"{verdicts['failed']} failures and {verdicts['passed']} successes from the user"
    )

    chosen: list[dict[str, Any]] = []
    for skill in env.skills.weakest(
        limit=cfg.max_topics_per_run, threshold=cfg.weak_score_threshold
    ):
        chosen.append(
            {
                "module": skill.module,
                "topic": skill.topic,
                "score": round(skill.score, 3),
                "trials": skill.trials,
                "demand": round(skill.demand, 3),
                "weakness": round(skill.weakness, 3),
                "reason": f"{skill.passes}/{skill.trials} verified attempts passed",
            }
        )

    # Nothing measured as weak: study what is in demand and unmeasured rather than
    # spending the night on a topic the user has never raised.
    if not chosen:
        for skill in sorted(env.skills.all(), key=lambda s: -s.demand):
            if skill.trials == 0 and skill.demand > 0:
                chosen.append(
                    {
                        "module": skill.module,
                        "topic": skill.topic,
                        "score": round(skill.score, 3),
                        "trials": 0,
                        "demand": round(skill.demand, 3),
                        "weakness": round(skill.weakness, 3),
                        "reason": "in demand, never measured",
                    }
                )
            if len(chosen) >= cfg.max_topics_per_run:
                break

    gaps = _gaps(env)
    unanswered = day.unanswered_questions(env.db, limit=20)
    payload = {
        "topics": chosen,
        "gaps": gaps,
        "unanswered": len(unanswered),
        "user_failures": verdicts["failed"],
        "user_successes": verdicts["passed"],
        "scored_attempts": scored_attempts,
        "skills": env.skills.summary(),
    }
    if env.dry_run:
        payload["dry_run"] = True
        return payload

    env.write_json("topics.json", payload)
    if chosen:
        env.events.emit(
            "sleep.reflected",
            "tonight's topics: " + ", ".join(f"{t['module']}/{t['topic']}" for t in chosen),
            run_id=env.run_id,
            topics=[f"{t['module']}/{t['topic']}" for t in chosen],
        )
    else:
        env.events.emit(
            "sleep.reflected",
            "no topic is measurably weak and in demand; nothing to study tonight",
            run_id=env.run_id,
        )
    if gaps:
        env.events.emit(
            "sleep.gaps",
            f"{len(gaps)} requests no module covered (a self-expansion trigger)",
            level="warning",
            run_id=env.run_id,
            gaps=len(gaps),
        )
    return payload
