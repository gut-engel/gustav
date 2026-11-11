"""Consolidation: turn the day into training material.

This is the step that decides what the system is allowed to learn from the day, so it
is deliberately conservative, and the order of priority is the user's:

1. **A correction is the answer.** `/correct` produces a training example weighted ten
   times a thumbs-up, marked verified, and nothing overrides it.
2. **A thumbs-up is an endorsement**, so the answer that was given becomes an example.
3. **A thumbs-down is a question, not an answer.** We know the reply was wrong; we do
   not know what the right one is. It becomes a *rewrite task* for `dreaming` to attempt
   under verification — never a training example on its own.
4. **Implicit feedback is a candidate.** It is counted and left for review; training on
   "the user rephrased the question" would teach noise.

Facts about the user are extracted from the day's messages by the deterministic
extractor, and — when a model is available — by the model as well, with every
model-proposed fact checked for being *grounded* in the message it came from. A fact the
model invented is worse than no fact: it gets trained on and then asserted.

Nothing here touches weights, so a crash is cheap: the step is `resume`, and the
checkpoint is how far through the day's exchanges it got.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from gustav.core.clock import RealClock
from gustav.inference.registry import load_registry
from gustav.learn import day
from gustav.learn.examples import Example
from gustav.learn.extract import facts_from_message, question_for
from gustav.learn.facts import Fact
from gustav.learn.grounding import grounded
from gustav.learn.jsonish import strings as json_strings
from gustav.learn.tasks import Task, TaskStore
from gustav.learn.topics import detect as detect_topics
from gustav.sleep.stages._base import StepEnv, sleep_step

FACT_SYSTEM = (
    "You extract durable facts about the user from a chat message. A durable fact is "
    "something that will still be true next week: a name, a preference, a tool, a "
    "machine, a project. Ignore anything transient, anything about the current task, "
    "and anything you are not told directly. Reply with a JSON array of short "
    "sentences written in the third person about 'the user', or [] if there is nothing."
)


def module_for(text: str, registry: Any, *, default: str = "general") -> str:
    topics = detect_topics(text, registry, limit=1)
    return topics[0].module if topics else default


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    registry = load_registry(env.paths.registry)
    tasks = TaskStore(env.db)
    cfg = env.config.sleep

    after = day.consolidated_through(env.db)
    since = None
    if after == 0:
        since = (RealClock().now() - timedelta(hours=cfg.lookback_hours)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    pairs = day.exchanges(env.db, after_turn_id=after, since=since, limit=cfg.max_examples_per_run)
    feedback = day.open_feedback(env.db)
    env.progress(f"{len(pairs)} exchanges, {len(feedback)} pieces of feedback")

    summary: dict[str, Any] = {
        "exchanges": len(pairs),
        "feedback": len(feedback),
        "corrections": 0,
        "endorsements": 0,
        "rewrite_tasks": 0,
        "implicit_pending": 0,
        "facts_new": 0,
        "facts_superseded": 0,
        "facts_from_model": 0,
        "examples": 0,
        "duplicates": 0,
        "heldout": 0,
        "model": False,
    }
    if env.dry_run:
        summary["dry_run"] = True
        return summary

    # --- the user's own verdicts ---------------------------------------
    # Each piece is marked consumed as it is handled, not in a batch at the end: a
    # preempted step has already stored those examples, and leaving the feedback open
    # would have the next night count them all over again.
    for _index, item in env.each(feedback, label="feedback", checkpoint_key="feedback_index"):
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        kind = item["kind"]
        if not item.get("trainable", 1):
            # The turn was forgotten or redacted: nothing to learn, nothing to keep.
            day.mark_feedback_consumed(env.db, [int(item["id"])], env.run_id)
            continue
        module = module_for(question or answer, registry)

        if kind == "correct" and question and (item.get("payload") or "").strip():
            added = env.examples.add(
                Example(
                    prompt=question,
                    completion=str(item["payload"]).strip(),
                    kind="correction",
                    module=module,
                    rejected=answer or None,
                    source="feedback",
                    source_ref=str(item["id"]),
                    verified=True,
                ),
                run_id=env.run_id,
            )
            summary["corrections"] += 1 if added else 0
        elif kind == "good" and question and answer:
            added = env.examples.add(
                Example(
                    prompt=question,
                    completion=answer,
                    kind="preference",
                    module=module,
                    source="feedback",
                    source_ref=str(item["id"]),
                    verified=True,
                ),
                run_id=env.run_id,
            )
            summary["endorsements"] += 1 if added else 0
        elif kind == "bad" and question and answer:
            topics = detect_topics(question, registry, limit=1)
            new_id = tasks.add(
                Task(
                    prompt=question,
                    kind="rewrite",
                    verifier={"rejected": answer, "question": question},
                    module=module,
                    topic=topics[0].topic if topics else None,
                    difficulty=0.8,
                    source="feedback",
                    source_ref=str(item["id"]),
                ),
                run_id=env.run_id,
            )
            summary["rewrite_tasks"] += 1 if new_id else 0
        elif kind == "implicit":
            summary["implicit_pending"] += 1
            continue  # left unconsumed on purpose: it is waiting for review
        day.mark_feedback_consumed(env.db, [int(item["id"])], env.run_id)

    # --- facts, and what the day was about ------------------------------
    model = env.model()
    summary["model"] = model.available
    fact_budget = cfg.max_facts_per_run

    for _index, exchange in env.each(pairs, label="exchange", checkpoint_key="exchange_index"):
        for topic in detect_topics(exchange.question, registry):
            env.skills.note_demand(topic.module, topic.topic)

        if fact_budget <= 0:
            continue
        candidates = facts_from_message(
            exchange.question, turn_id=exchange.question_turn_id
        )
        if model.available and not candidates and len(exchange.question) > 24:
            answer = model.ask(exchange.question, system=FACT_SYSTEM, max_tokens=256)
            for statement in json_strings(answer or "")[:3]:
                if len(statement) < 8 or not grounded(statement, exchange.question):
                    continue
                summary["facts_from_model"] += 1
                candidates.append(
                    Fact(
                        text=statement.strip(),
                        subject=None,
                        confidence=0.55,
                        source="model",
                        source_turn_id=exchange.question_turn_id,
                    )
                )
        for fact in candidates[:fact_budget]:
            change = env.facts.add(fact, run_id=env.run_id)
            if change is None:
                continue
            fact_budget -= 1
            summary["facts_new"] += 1 if change.is_new else 0
            summary["facts_superseded"] += len(change.superseded)
            # A fact is only worth training on once it is stated with some confidence.
            if fact.confidence >= 0.7:
                env.examples.add(
                    Example(
                        prompt=question_for(fact),
                        completion=fact.text,
                        kind="fact",
                        module=fact.module,
                        source="turn",
                        source_ref=str(fact.source_turn_id or ""),
                        verified=fact.source != "model",
                    ),
                    run_id=env.run_id,
                )

    # --- bookkeeping ----------------------------------------------------
    if pairs:
        day.set_consolidated_through(env.db, pairs[-1].question_turn_id)
    else:
        day.set_consolidated_through(env.db, day.last_turn_id(env.db))

    counts = env.examples.counts(run_id=env.run_id)
    summary["examples"] = sum(counts.values())
    summary["by_kind"] = counts
    summary["heldout"] = len(env.examples.heldout(limit=10_000))
    env.write_json("summary.json", summary)
    env.events.emit(
        "sleep.consolidated",
        f"{summary['examples']} examples, {summary['facts_new']} new facts "
        f"from {len(pairs)} exchanges",
        run_id=env.run_id,
        **{k: summary[k] for k in ("corrections", "endorsements", "rewrite_tasks")},
    )
    return summary
