"""Study: turn documentation into cited answers and verifiable exercises.

Two products, and the difference between them is the whole point of this step.

**Cited answers.** A question and an answer drawn from one section of one document,
stored with the URL and heading they came from. Every model-written answer is checked
for being *grounded* in the excerpt — its content words have to actually appear in the
source — because a fluent answer that the document does not support is exactly the kind
of training data that produces a confident, wrong model. With no model available the
step still works: the excerpt itself is the answer, which is extractive and therefore
trivially grounded, just less varied.

**Exercises.** The model is asked for a small task *with a reference solution and
tests*, and then the task is admitted only if the sandbox agrees twice over:

* the reference solution passes the tests — otherwise the task is incoherent, and
* the tests fail without the solution — otherwise they test nothing.

That second check is the one that matters. Self-generated exercises are worthless
without it: a model that writes ``assert True`` has produced a task that everything
passes, and a night of practice against it measures nothing while looking productive.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gustav.learn.chunk import best_chunks, chunks
from gustav.learn.docs import DocumentStore
from gustav.learn.examples import Example
from gustav.learn.grounding import grounded
from gustav.learn.jsonish import loads as json_loads
from gustav.learn.jsonish import objects as json_objects
from gustav.learn.tasks import Task, TaskStore
from gustav.sandbox import Sandbox
from gustav.sleep.stages._base import StepEnv, sleep_step

MAX_CHUNKS_PER_DOC = 4
MAX_PAIRS_PER_CHUNK = 3

QA_SYSTEM = (
    "You turn documentation into question-and-answer pairs for training a local "
    "assistant. Every answer must be supported by the excerpt you are given and must "
    "add nothing that is not in it. Questions must be answerable without seeing the "
    "excerpt. Reply with a JSON array of objects with keys \"question\" and \"answer\", "
    "and nothing else."
)

TASK_SYSTEM = (
    "You write small self-contained Python exercises for a local assistant to practise "
    "on. Reply with one JSON object and nothing else, with keys: \"prompt\" (what to "
    "implement, naming the function exactly), \"setup\" (imports or fixtures, may be "
    "empty), \"solution\" (the reference implementation) and \"tests\" (assert "
    "statements that fail if the implementation is wrong). The tests must not pass "
    "unless the function is implemented correctly. No input(), no network, no files."
)

_FUNC = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.MULTILINE)


def _stub_for(solution: str) -> str:
    """The same functions, defined and doing nothing.

    Used to prove the tests actually test something: with every function replaced by a
    stub, the tests have to fail.
    """
    names = _FUNC.findall(solution)
    if not names:
        return "pass"
    return "\n".join(f"def {name}(*args, **kwargs):\n    return None" for name in names)


def _read_document(row: dict[str, Any]) -> str:
    path = Path(row.get("path") or "")
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8", errors="replace")
    _, _, body = raw.partition("\n")  # the cache file starts with "# title"
    return body


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    cfg = env.config.sleep
    docs = DocumentStore(env.db)
    tasks = TaskStore(env.db)
    model = env.model()
    sandbox = Sandbox(
        python=env.config.sandbox.python,
        timeout_seconds=env.config.sandbox.timeout_seconds,
        memory_limit_mb=env.config.sandbox.memory_limit_mb,
        max_output_bytes=env.config.sandbox.max_output_bytes,
        workdir=env.paths.tmp,
        isolate_env=env.config.sandbox.isolate_env,
    )

    pending = docs.unstudied(limit=env.config.fetcher.max_documents_per_run, run_id=env.run_id)
    result: dict[str, Any] = {
        "documents": len(pending),
        "chunks": 0,
        "examples": 0,
        "rejected_ungrounded": 0,
        "duplicates": 0,
        "tasks_proposed": 0,
        "tasks_admitted": 0,
        "tasks_rejected": {"unparseable": 0, "solution_failed": 0, "tests_vacuous": 0},
        "model": model.available,
    }
    if not pending:
        result["reason"] = "nothing new to study"
        return result
    if env.dry_run:
        result["dry_run"] = True
        return result

    for _index, row in env.each(pending, label="document", checkpoint_key="doc_index"):
        text = _read_document(row)
        if not text.strip():
            docs.mark_studied(int(row["id"]), env.run_id)
            continue
        topic = row.get("topic") or ""
        pieces = best_chunks(chunks(text), topic, limit=MAX_CHUNKS_PER_DOC)
        result["chunks"] += len(pieces)
        module = _module_for_topic(env, topic)
        title = row.get("title") or row["url"]

        for chunk in pieces:
            env.raise_if_stopping()
            citation = chunk.citation(row["url"])
            pairs = _pairs_for(env, model, title, row["url"], chunk, result)
            for question, answer in pairs:
                added = env.examples.add(
                    Example(
                        prompt=question,
                        completion=answer,
                        kind="study",
                        module=module,
                        source="document",
                        source_ref=str(row["id"]),
                        citation=citation,
                        # Grounded in a document is not the same as verified: no test
                        # and no user has agreed with it. The gate decides.
                        verified=False,
                    ),
                    run_id=env.run_id,
                )
                if added:
                    result["examples"] += 1
                else:
                    result["duplicates"] += 1

        if model.available and pieces:
            admitted = _exercise_from(
                env, model, sandbox, tasks, pieces[0], row, module, topic, result
            )
            result["tasks_admitted"] += 1 if admitted else 0

        docs.mark_studied(int(row["id"]), env.run_id)
        if result["examples"] >= cfg.max_examples_per_run:
            result["reason"] = "hit sleep.max_examples_per_run"
            break

    env.write_json("study.json", result)
    env.events.emit(
        "sleep.studied",
        f"{result['examples']} cited examples from {result['documents']} documents, "
        f"{result['tasks_admitted']} exercises admitted",
        run_id=env.run_id,
        examples=result["examples"],
        tasks=result["tasks_admitted"],
    )
    return result


def _module_for_topic(env: StepEnv, topic: str) -> str:
    row = env.db.query_one(
        "SELECT module FROM skills WHERE topic = ? ORDER BY demand DESC LIMIT 1", (topic,)
    )
    return row["module"] if row else "general"


def _pairs_for(
    env: StepEnv, model: Any, title: str, url: str, chunk: Any, result: dict[str, Any]
) -> list[tuple[str, str]]:
    if not model.available:
        # Extractive fallback: the section answers the question about itself.
        heading = chunk.heading or title
        question = f"What does the documentation for {title} say about {heading}?"
        return [(question, chunk.text.strip()[:1200])]

    prompt = (
        f"Source: {title} ({url})\n"
        f"Section: {chunk.heading or '(none)'}\n\n"
        f"Excerpt:\n{chunk.text}\n\n"
        f"Write up to {MAX_PAIRS_PER_CHUNK} question/answer pairs."
    )
    answer = model.ask(prompt, system=QA_SYSTEM, max_tokens=env.config.sleep.max_new_tokens)
    pairs: list[tuple[str, str]] = []
    for item in json_objects(answer or "")[:MAX_PAIRS_PER_CHUNK]:
        question = str(item.get("question") or "").strip()
        reply = str(item.get("answer") or "").strip()
        if len(question) < 8 or len(reply) < 8:
            continue
        if not grounded(reply, chunk.text):
            result["rejected_ungrounded"] += 1
            continue
        pairs.append((question, reply))
    return pairs


def _exercise_from(
    env: StepEnv,
    model: Any,
    sandbox: Sandbox,
    tasks: TaskStore,
    chunk: Any,
    row: dict[str, Any],
    module: str,
    topic: str,
    result: dict[str, Any],
) -> bool:
    env.progress(f"writing an exercise from {row['url']}")
    answer = model.ask(
        f"Documentation excerpt:\n{chunk.text}\n\n"
        f"Write one small exercise that practises what this excerpt teaches.",
        system=TASK_SYSTEM,
        max_tokens=max(env.config.sleep.max_new_tokens, 512),
    )
    result["tasks_proposed"] += 1
    payload = json_loads(answer or "")
    if not isinstance(payload, dict):
        result["tasks_rejected"]["unparseable"] += 1
        return False
    prompt = str(payload.get("prompt") or "").strip()
    setup = str(payload.get("setup") or "").strip()
    solution = str(payload.get("solution") or "").strip()
    tests = str(payload.get("tests") or "").strip()
    if not (prompt and solution and "assert" in tests):
        result["tasks_rejected"]["unparseable"] += 1
        return False

    reference = sandbox.run_python(solution, setup=setup, tests=tests)
    if not reference.passed:
        env.log(f"discarded an exercise: its own solution fails ({reference.detail})")
        result["tasks_rejected"]["solution_failed"] += 1
        return False

    # The test that keeps this honest: with the functions stubbed out, the tests must
    # fail. Otherwise they assert nothing and every attempt would "pass".
    vacuous = sandbox.run_python(_stub_for(solution), setup=setup, tests=tests)
    if vacuous.passed:
        env.log("discarded an exercise: its tests pass without a solution", level="warning")
        result["tasks_rejected"]["tests_vacuous"] += 1
        return False

    new_id = tasks.add(
        Task(
            prompt=prompt,
            kind="python",
            verifier={"setup": setup, "tests": tests, "reference": solution},
            module=module,
            topic=topic or None,
            difficulty=0.5,
            source="document",
            source_ref=row["url"],
            document_id=int(row["id"]),
        ),
        run_id=env.run_id,
    )
    return new_id is not None
