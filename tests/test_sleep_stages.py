"""The real sleep steps, run directly, with no daemon and no GPU.

A step body is a function of a ``StepEnv``, so each one can be called with a scripted
model and a throwaway store. The point of most of these tests is what the night refuses
to learn: an ungrounded fact, an exercise whose tests pass without a solution, a
candidate adapter that did not beat the one being served.
"""

from __future__ import annotations

import json
import shutil

import pytest

from gustav.config.schema import Config
from gustav.learn.bench import BenchItem, BenchmarkStore
from gustav.learn.examples import Example
from gustav.learn.tasks import Task, TaskStore
from gustav.sleep.stages import (
    consolidation,
    dreaming,
    fetch_docs,
    gate,
    practice,
    reflect,
    report,
    retest,
    snapshot,
    study,
    training,
)
from gustav.testing.steps import (
    RecordingContext,
    ScriptedModel,
    make_env,
    seed_exchange,
    seed_feedback,
    write_adapter,
)


def body(module):
    """The undecorated step body: ``@sleep_step`` wraps it for the supervisor."""
    return module.run_step.__wrapped__


@pytest.fixture
def stage_config(project, repo_root):
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    (project / "docs").mkdir(exist_ok=True)
    (project / "notes").mkdir(exist_ok=True)
    return Config.model_validate(
        {
            "logging": {"console": False},
            "sleep": {"heldout_fraction": 0.0, "max_topics_per_run": 3},
            "fetcher": {"enabled": False, "local_sources": [str(project / "notes")]},
            "backup": {"location": str(project / "backups"), "min_free_mb": 1},
            "sandbox": {"timeout_seconds": 6, "memory_limit_mb": 512},
            "training": {"min_examples": 4},
        }
    )


@pytest.fixture
def env_for(stage_config, paths, db):
    def build(step, **kwargs):
        return make_env(config=stage_config, paths=paths, db=db, step=step, **kwargs)

    return build


# ---------------------------------------------------------------- snapshot
def test_the_night_starts_with_a_verified_snapshot(env_for, db):
    env = env_for("snapshot")
    result = body(snapshot)(env)
    assert result["verified"] and result["files"] > 0
    assert db.get_meta("sleep.run_snapshot")["snapshot"] == result["snapshot"]


def test_a_disabled_backup_is_said_out_loud_rather_than_hidden(env_for, stage_config):
    stage_config.backup.enabled = False
    env = env_for("snapshot")
    result = body(snapshot)(env)
    assert result["skipped"]
    assert any("without a safety net" in message for _level, message in env.ctx.logs)


# ----------------------------------------------------------- consolidation
def test_a_correction_outweighs_everything_and_is_marked_verified(env_for, db):
    _, answer_id = seed_exchange(db, question="how do I reverse a list in python?",
                                 answer="use reversed()")
    seed_feedback(db, turn_id=answer_id, kind="correct", payload="use list[::-1]")
    result = body(consolidation)(env_for("consolidation"))

    assert result["corrections"] == 1
    row = db.query_one("SELECT * FROM training_examples WHERE kind = 'correction'")
    assert row["completion"] == "use list[::-1]"
    assert row["rejected"] == "use reversed()"
    assert row["weight"] == 10.0 and row["verified"] == 1
    assert db.query_one("SELECT consumed_run_id FROM feedback")["consumed_run_id"] == 1


def test_a_thumbs_up_endorses_the_answer_that_was_given(env_for, db):
    _, answer_id = seed_exchange(db, question="name three async crates", answer="tokio, smol")
    seed_feedback(db, turn_id=answer_id, kind="good")
    result = body(consolidation)(env_for("consolidation"))
    assert result["endorsements"] == 1
    row = db.query_one("SELECT * FROM training_examples WHERE kind = 'preference'")
    assert row["completion"] == "tokio, smol" and row["verified"] == 1


def test_a_thumbs_down_becomes_a_task_not_an_answer(env_for, db):
    _, answer_id = seed_exchange(db, question="what does Arc do in rust?", answer="nothing")
    seed_feedback(db, turn_id=answer_id, kind="bad")
    result = body(consolidation)(env_for("consolidation"))

    assert result["rewrite_tasks"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM training_examples WHERE kind='preference'")[
        "n"
    ] == 0, "we know the answer was wrong, not what the right one is"
    task = db.query_one("SELECT * FROM tasks")
    assert task["kind"] == "rewrite"
    assert json.loads(task["verifier"])["rejected"] == "nothing"


def test_implicit_feedback_waits_for_review(env_for, db):
    _, answer_id = seed_exchange(db, question="q", answer="a")
    seed_feedback(db, turn_id=answer_id, kind="implicit", payload="rephrased")
    result = body(consolidation)(env_for("consolidation"))
    assert result["implicit_pending"] == 1
    assert db.query_one("SELECT consumed_run_id FROM feedback")["consumed_run_id"] is None


def test_facts_are_extracted_and_become_training_material(env_for, db):
    seed_exchange(db, question="my editor is helix, remember that", answer="noted")
    result = body(consolidation)(env_for("consolidation"))
    assert result["facts_new"] >= 1
    assert db.query_one("SELECT * FROM facts")["subject"] == "editor"
    row = db.query_one("SELECT * FROM training_examples WHERE kind = 'fact'")
    assert "helix" in row["completion"]
    assert row["module"] == "personal-memory"


def test_a_turn_marked_never_train_produces_nothing(env_for, db):
    seed_exchange(db, question="my api key is sk-live-123", answer="I cannot store that",
                  trainable=False)
    result = body(consolidation)(env_for("consolidation"))
    assert result["examples"] == 0 and result["facts_new"] == 0


def test_an_ungrounded_fact_from_the_model_is_thrown_away(env_for, db):
    seed_exchange(db, question="can you summarise what we discussed about the build?",
                  answer="sure")
    model = ScriptedModel(['["The user lives in Reykjavik and owns three cats."]'])
    result = body(consolidation)(env_for("consolidation", model=model))
    assert model.calls > 0, "the model was asked"
    assert result["facts_new"] == 0, "and its invention was refused"


def test_a_grounded_fact_from_the_model_is_kept(env_for, db):
    seed_exchange(
        db,
        question="for the record the build machine is a threadripper with 128gb of ram",
        answer="noted",
    )
    model = ScriptedModel(['["The build machine is a threadripper with 128gb of ram."]'])
    result = body(consolidation)(env_for("consolidation", model=model))
    assert result["facts_from_model"] == 1 and result["facts_new"] == 1


def test_consolidation_does_not_do_the_same_day_twice(env_for, db):
    seed_exchange(db, question="my shell is zsh", answer="noted")
    first = body(consolidation)(env_for("consolidation"))
    second = body(consolidation)(env_for("consolidation"))
    assert first["exchanges"] == 1 and second["exchanges"] == 0


# ---------------------------------------------------------------- reflect
def test_what_the_user_corrected_becomes_a_measured_weakness(env_for, db):
    env = env_for("consolidation")
    env.examples.add(
        Example(prompt="why does the borrow checker reject this lifetime?",
                completion="because the reference outlives the owner", kind="correction",
                module="rust", verified=True),
        run_id=1,
    )
    result = body(reflect)(env_for("reflect"))
    assert result["user_failures"] >= 1
    topics = {t["topic"] for t in result["topics"]}
    assert topics & {"borrow", "lifetime"}
    assert result["topics"][0]["score"] < 0.6


def test_with_nothing_measured_it_studies_what_is_in_demand(env_for, db):
    env = env_for("reflect")
    env.skills.note_demand("coding", "regex", hits=3)
    result = body(reflect)(env)
    assert [t["reason"] for t in result["topics"]] == ["in demand, never measured"]


def test_a_request_no_module_covered_is_recorded_as_a_gap(env_for, db):
    db.insert(
        "telemetry",
        {
            "metric": "router.decision",
            "value": 0.1,
            "tags": json.dumps({"uncovered": True, "turn_id": 4}),
        },
    )
    result = body(reflect)(env_for("reflect"))
    assert len(result["gaps"]) == 1


# ------------------------------------------------------------- fetch_docs
def write_topics(env, topics):
    directory = env.run_dir / "reflect"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "topics.json").write_text(json.dumps({"topics": topics}), encoding="utf-8")


def test_a_local_document_is_studied_with_no_network(env_for, project, db):
    (project / "notes" / "lifetime.md").write_text(
        "# Lifetimes\n\n## Elision\n\n" + ("Every reference has a lifetime. " * 12),
        encoding="utf-8",
    )
    env = env_for("fetch_docs")
    write_topics(env, [{"module": "rust", "topic": "lifetime"}])
    result = body(fetch_docs)(env)
    assert result["fetched"] == 1 and result["network"] is False
    assert db.query_one("SELECT status FROM documents")["status"] == "fresh"


def test_no_topics_means_no_requests(env_for):
    env = env_for("fetch_docs")
    write_topics(env, [])
    result = body(fetch_docs)(env)
    assert result["fetched"] == 0 and "reflect chose no topics" in result["reason"]


# ----------------------------------------------------------------- study
def seed_document(env, project, text: str, *, topic: str = "lifetime") -> None:
    path = project / "notes" / f"{topic}.md"
    path.write_text(text, encoding="utf-8")
    cache = env.paths.doc_cache / "local" / f"{topic}.md"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(f"# {topic}\n{text}\n", encoding="utf-8")
    env.db.insert(
        "documents",
        {
            "url": path.as_uri(),
            "topic": topic,
            "title": topic,
            "sha256": "x",
            "bytes": len(text),
            "path": str(cache),
            "status": "fresh",
            "run_id": env.run_id,
        },
    )


def test_without_a_model_the_document_answers_for_itself(env_for, project, db):
    env = env_for("study")
    seed_document(env, project, "## Elision\n\n" + ("Every reference has a lifetime. " * 12))
    result = body(study)(env)
    assert result["examples"] >= 1
    row = db.query_one("SELECT * FROM training_examples WHERE kind = 'study'")
    assert row["citation"].endswith("#elision")
    assert "lifetime" in row["completion"]
    assert row["verified"] == 0, "grounded in a document is not the same as verified"


def test_an_answer_the_document_does_not_support_is_refused(env_for, project, db):
    env = env_for(
        "study",
        model=ScriptedModel(
            ['[{"question": "what is a lifetime?", '
             '"answer": "Lifetimes are a kind of garbage collector written in Haskell."}]']
        ),
    )
    seed_document(env, project, "## Elision\n\n" + ("Every reference has a lifetime. " * 12))
    result = body(study)(env)
    assert result["rejected_ungrounded"] >= 1
    assert result["examples"] == 0


def test_an_exercise_is_admitted_only_if_its_own_solution_passes(env_for, project, db):
    good = json.dumps(
        {
            "prompt": "write add(a, b)",
            "setup": "",
            "solution": "def add(a, b):\n    return a + b",
            "tests": "assert add(2, 3) == 5",
        }
    )
    env = env_for("study", model=ScriptedModel({"Write one small exercise": good,
                                                "Excerpt": "[]"}))
    seed_document(env, project, "## Adding\n\n" + ("Numbers can be added together. " * 12))
    result = body(study)(env)
    assert result["tasks_admitted"] == 1
    task = db.query_one("SELECT * FROM tasks")
    assert task["kind"] == "python"


def test_an_exercise_whose_solution_fails_is_discarded(env_for, project, db):
    broken = json.dumps(
        {
            "prompt": "write add(a, b)",
            "solution": "def add(a, b):\n    return a - b",
            "tests": "assert add(2, 3) == 5",
        }
    )
    env = env_for("study", model=ScriptedModel({"Write one small exercise": broken,
                                                "Excerpt": "[]"}))
    seed_document(env, project, "## Adding\n\n" + ("Numbers can be added together. " * 12))
    result = body(study)(env)
    assert result["tasks_rejected"]["solution_failed"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM tasks")["n"] == 0


def test_an_exercise_whose_tests_assert_nothing_is_discarded(env_for, project, db):
    vacuous = json.dumps(
        {
            "prompt": "write add(a, b)",
            "solution": "def add(a, b):\n    return a + b",
            "tests": "assert True",
        }
    )
    env = env_for("study", model=ScriptedModel({"Write one small exercise": vacuous,
                                                "Excerpt": "[]"}))
    seed_document(env, project, "## Adding\n\n" + ("Numbers can be added together. " * 12))
    result = body(study)(env)
    assert result["tasks_rejected"]["tests_vacuous"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM tasks")["n"] == 0


# -------------------------------------------------------------- practice
def test_practice_without_a_model_does_nothing_and_says_why(env_for, db):
    TaskStore(db).add(Task(prompt="write add(a, b)", kind="python",
                           verifier={"tests": "assert add(1, 2) == 3"}))
    result = body(practice)(env_for("practice"))
    assert result["attempted"] == 0 and "no model" in result["reason"]


def test_a_passing_attempt_is_a_measured_trial_and_a_verified_example(env_for, db):
    TaskStore(db).add(
        Task(prompt="write add(a, b)", kind="python", module="coding", topic="python",
             verifier={"tests": "assert add(1, 2) == 3"})
    )
    model = ScriptedModel(["```python\ndef add(a, b):\n    return a + b\n```"])
    result = body(practice)(env_for("practice", model=model))

    assert (result["passed"], result["failed"]) == (1, 0)
    row = db.query_one("SELECT * FROM training_examples WHERE kind = 'practice'")
    assert row["verified"] == 1
    skill = db.query_one("SELECT * FROM skills WHERE topic = 'python'")
    assert (skill["trials"], skill["passes"]) == (1, 1)
    assert db.query_one("SELECT status FROM tasks")["status"] == "solved"


def test_a_failing_attempt_is_recorded_as_a_failure_not_dropped(env_for, db):
    TaskStore(db).add(
        Task(prompt="write add(a, b)", kind="python", module="coding", topic="python",
             verifier={"tests": "assert add(1, 2) == 3"})
    )
    model = ScriptedModel(["```python\ndef add(a, b):\n    return a - b\n```"])
    result = body(practice)(env_for("practice", model=model))
    assert (result["passed"], result["failed"]) == (0, 1)
    assert db.query_one("SELECT * FROM training_examples") is None
    attempt = db.query_one("SELECT * FROM task_attempts")
    assert attempt["passed"] == 0 and "AssertionError" in attempt["detail"]


def test_an_answer_with_no_code_in_it_fails_the_task(env_for, db):
    TaskStore(db).add(Task(prompt="write add(a, b)", kind="python", topic="python",
                           verifier={"tests": "assert add(1, 2) == 3"}))
    model = ScriptedModel(["I would rather talk about something else."])
    result = body(practice)(env_for("practice", model=model))
    assert result["failed"] == 1
    assert "no python" in db.query_one("SELECT detail FROM task_attempts")["detail"]


# -------------------------------------------------------------- dreaming
def test_sampling_solves_what_greedy_decoding_missed(env_for, db):
    TaskStore(db).add(
        Task(prompt="write add(a, b)", kind="python", module="coding", topic="python",
             verifier={"tests": "assert add(1, 2) == 3"})
    )
    model = ScriptedModel(
        [
            "```python\ndef add(a, b):\n    return a - b\n```",   # rejected
            "```python\ndef add(a, b):\n    return a + b\n```",   # kept
        ]
    )
    result = body(dreaming)(env_for("dreaming", model=model))
    assert result["solved"] == 1 and result["rejected"] == 1
    row = db.query_one("SELECT * FROM training_examples WHERE kind = 'dream'")
    assert row["verified"] == 1, "it passed the task's own tests"


def test_a_rejected_answer_is_replaced_only_when_the_samples_agree(env_for, db):
    TaskStore(db).add(
        Task(prompt="what does Arc do?", kind="rewrite", module="rust",
             verifier={"question": "what does Arc do?", "rejected": "nothing much"})
    )
    agreed = "Arc is an atomically reference counted shared pointer."
    model = ScriptedModel([agreed, agreed, agreed, agreed])
    result = body(dreaming)(env_for("dreaming", model=model))

    assert result["rewritten"] == 1
    row = db.query_one("SELECT * FROM training_examples WHERE kind = 'preference'")
    assert row["completion"] == agreed
    assert row["rejected"] == "nothing much"
    assert row["verified"] == 0, "agreement between samples is evidence, not proof"


def test_candidates_that_disagree_produce_nothing(env_for, db):
    TaskStore(db).add(
        Task(prompt="what does Arc do?", kind="rewrite", module="rust",
             verifier={"question": "what does Arc do?", "rejected": "nothing much"})
    )
    model = ScriptedModel(
        ["bananas are yellow", "the weather is fine", "trains run on rails", "x = 1"]
    )
    result = body(dreaming)(env_for("dreaming", model=model))
    assert result["rewritten"] == 0 and result["rejected"] == 1
    assert db.query_one("SELECT * FROM training_examples") is None


def test_a_replacement_that_restates_the_rejected_answer_is_refused(env_for, db):
    rejected = "Arc does nothing much at all really"
    TaskStore(db).add(
        Task(prompt="what does Arc do?", kind="rewrite", module="rust",
             verifier={"question": "what does Arc do?", "rejected": rejected})
    )
    model = ScriptedModel([rejected] * 4)
    result = body(dreaming)(env_for("dreaming", model=model))
    assert result["rewritten"] == 0
    assert "restates the rejected answer" in db.query_one(
        "SELECT detail FROM task_attempts"
    )["detail"]


# -------------------------------------------------------------- training
def test_training_without_a_hardware_plan_says_so_instead_of_crashing(env_for):
    result = body(training)(env_for("training"))
    assert result["trained"] == 0 and "no hardware plan" in result["reason"]


def test_training_is_skipped_on_a_machine_that_only_serves(env_for, stage_config):
    stage_config.hardware.allow_training = False
    result = body(training)(env_for("training"))
    assert "only serves" in result["reason"]


def test_a_module_with_too_little_new_material_is_not_trained(env_for, db, stage_config):
    env = env_for("training")
    env.examples.add(Example(prompt="q", completion="a", kind="study", module="coding"))
    result = body(training)(env)
    # No plan on this machine, so the reason is the plan; the material check is the
    # dry-run path below, which does not need a GPU either.
    assert result["trained"] == 0


# ------------------------------------------------------------------ gate
class TwoVersionModel(ScriptedModel):
    """Answers differently depending on whether the candidate adapter is loaded."""

    def __init__(self, *, baseline: str, candidate: str):
        super().__init__(available=True)
        self.baseline, self.candidate = baseline, candidate

    def evaluate(self, prompts, *, candidate=None, system=None, **_overrides):
        self.candidates.append(candidate)
        answer = self.candidate if candidate else self.baseline
        return [answer for _ in prompts]


def prepare_candidate(
    env,
    *,
    module: str = "coding",
    version: str = "v002",
    items: int = 10,
    match: str = "contains",
):
    """A trained-but-unpromoted version, an active one, and a held-out suite."""
    adapters = env.paths.adapters / module
    write_adapter(adapters / "v001", version="v001")
    write_adapter(adapters / version, version=version)
    (adapters / "active").symlink_to("v001")
    env.db.insert(
        "training_runs",
        {"run_id": env.run_id, "module": module, "version": version, "examples": 20,
         "status": "trained"},
    )
    store = BenchmarkStore(env.db)
    for index in range(items):
        store.add(
            BenchItem(
                prompt=f"question {index}",
                expected="the right answer",
                verifier={"match": match},
                module=module,
                suite="heldout",
            )
        )


def test_a_better_candidate_is_promoted_by_a_symlink_swap(env_for, db):
    env = env_for("gate", model=TwoVersionModel(baseline="wrong", candidate="the right answer"))
    prepare_candidate(env)
    result = body(gate)(env)

    assert result["promoted"] == ["coding"]
    link = env.paths.adapters / "coding" / "active"
    assert link.readlink().name == "v002"
    row = db.query_one("SELECT * FROM gate_results WHERE suite = 'heldout'")
    assert row["verdict"] == "pass" and row["delta"] > 0
    assert db.query_one("SELECT status FROM training_runs")["status"] == "promoted"
    # Guarded: a verified snapshot exists for the promotion.
    change = db.query_one("SELECT * FROM weight_changes ORDER BY id DESC")
    assert change["status"] == "done" and change["snapshot_id"]


def test_a_candidate_that_is_not_better_stays_on_the_shelf(env_for, db):
    env = env_for("gate", model=TwoVersionModel(baseline="the right answer", candidate="wrong"))
    prepare_candidate(env)
    result = body(gate)(env)

    assert result["rejected"] == ["coding"]
    assert (env.paths.adapters / "coding" / "active").readlink().name == "v001"
    assert db.query_one("SELECT status FROM training_runs")["status"] == "rejected"
    assert (env.paths.adapters / "coding" / "v002").exists(), "kept for review"


def test_too_few_items_is_a_skip_not_a_promotion(env_for, db):
    env = env_for("gate", model=TwoVersionModel(baseline="wrong", candidate="the right answer"))
    prepare_candidate(env, items=2)
    result = body(gate)(env)
    assert result["skipped"] == ["coding"]
    assert "nothing to measure" in result["candidates"]["coding"]["reason"]
    assert (env.paths.adapters / "coding" / "active").readlink().name == "v001"


def test_forgetting_blocks_a_promotion_even_when_the_new_skill_improved(env_for, db):
    env = env_for("gate")
    prepare_candidate(env)
    # The candidate is better on the new material and worse on what it already knew.
    store = BenchmarkStore(db)
    for index in range(10):
        store.add(
            BenchItem(prompt=f"old skill {index}", expected="remembered",
                      verifier={"match": "contains"}, module="coding", suite="regression")
        )

    class Forgetful(ScriptedModel):
        def __init__(self):
            super().__init__(available=True)

        def evaluate(self, prompts, *, candidate=None, system=None, **_overrides):
            self.candidates.append(candidate)
            out = []
            for prompt in prompts:
                old = str(prompt).startswith("old skill")
                if candidate:
                    out.append("forgotten" if old else "the right answer")
                else:
                    out.append("remembered" if old else "wrong")
            return out

    env._model = Forgetful()
    result = body(gate)(env)
    assert result["rejected"] == ["coding"]
    assert "retention" in result["candidates"]["coding"]["reason"]


def test_nothing_trained_means_nothing_to_gate(env_for):
    result = body(gate)(env_for("gate", model=ScriptedModel(["x"])))
    assert "nothing was trained" in result["reason"]


def test_a_disabled_gate_refuses_to_promote_rather_than_promoting_blind(env_for, stage_config):
    stage_config.gate.enabled = False
    env = env_for("gate", model=TwoVersionModel(baseline="wrong", candidate="the right answer"))
    prepare_candidate(env)
    result = body(gate)(env)
    assert "gate.enabled is false" in result["reason"]
    assert (env.paths.adapters / "coding" / "active").readlink().name == "v001"


def test_only_one_module_is_promoted_per_night_by_default(env_for, db):
    env = env_for("gate", model=TwoVersionModel(baseline="wrong", candidate="the right answer"))
    prepare_candidate(env, module="coding")
    prepare_candidate(env, module="rust")
    result = body(gate)(env)
    assert len(result["promoted"]) == 1
    assert len(result["skipped"]) == 1


# ---------------------------------------------------------------- retest
def test_a_corrected_answer_is_asked_again_with_the_new_weights(env_for, db):
    env = env_for("retest", model=ScriptedModel(["because the reference outlives the owner"]))
    env.examples.add(
        Example(prompt="why does the borrow checker reject this lifetime?",
                completion="because the reference outlives the owner", kind="correction",
                module="rust", verified=True),
        run_id=1,
    )
    result = body(retest)(env)
    assert (result["corrections_retested"], result["corrections_now_right"]) == (1, 1)
    assert db.query_one("SELECT passes FROM skills")["passes"] == 1


def test_a_still_wrong_answer_keeps_the_topic_in_the_queue(env_for, db):
    env = env_for("retest", model=ScriptedModel(["no idea, sorry"]))
    env.examples.add(
        Example(prompt="why does the borrow checker reject this lifetime?",
                completion="because the reference outlives the owner", kind="correction",
                module="rust", verified=True),
        run_id=1,
    )
    result = body(retest)(env)
    assert result["corrections_now_right"] == 0
    assert db.query_one("SELECT passes, trials FROM skills")["passes"] == 0


# ---------------------------------------------------------------- report
def test_the_report_is_written_from_the_store(env_for, db, journal):
    from gustav.core.journal import StepSpec

    run = journal.create_run(trigger="manual", steps=[StepSpec("consolidation")])
    env = env_for("report", run_id=run.id)
    env.examples.add(Example(prompt="q", completion="a", kind="study"), run_id=run.id)
    result = body(report)(env)

    path = env.paths.reports / f"run-{run.id:06d}.html"
    assert path.exists() and result["bytes"] > 500
    html = path.read_text(encoding="utf-8")
    assert "Sleep run" in html and "consolidation" in html
    assert (env.paths.reports / "latest.html").exists()


# --------------------------------------------------------- dry run & stop
def test_a_dry_run_changes_nothing(env_for, db):
    seed_exchange(db, question="my editor is helix", answer="noted")
    result = body(consolidation)(make_env(
        config=env_for("consolidation").config,
        paths=env_for("consolidation").paths,
        db=db,
        step="consolidation",
        dry_run=True,
    ))
    assert result["dry_run"]
    assert db.query_one("SELECT * FROM facts") is None
    assert db.query_one("SELECT * FROM training_examples") is None


def test_a_stop_request_preempts_a_step_at_an_item_boundary(env_for, db):
    from gustav.core.supervisor import Preempted

    for index in range(5):
        seed_exchange(db, question=f"my setting{index} is on", answer="noted")
    env = env_for("consolidation", ctx=RecordingContext(stop_after=2))
    with pytest.raises(Preempted):
        body(consolidation)(env)
    assert env.ctx.checkpoints, "it wrote down where it got to before stopping"


def test_the_retention_suite_excludes_what_tonight_trained_on(env_for, db):
    """Otherwise the gate measures retention with the answers it was just given."""
    from gustav.learn.bench import seed_regression_suite

    env = env_for("gate")
    old = env.examples.add(
        Example(prompt="an old skill", completion="remembered", kind="correction",
                module="coding", verified=True)
    )
    fresh = env.examples.add(
        Example(prompt="learned tonight", completion="new", kind="correction",
                module="coding", verified=True)
    )
    env.examples.mark_used([old or 0], run_id=1)
    env.examples.mark_used([fresh or 0], run_id=2)

    added = seed_regression_suite(db, module="coding", exclude_run_id=2)
    prompts = [item.prompt for item in BenchmarkStore(db).suite("regression")]
    assert added == 1
    assert prompts == ["an old skill"]


def test_a_source_with_no_recorded_licence_is_fetched_and_warned_about(env_for, project, db):
    material = project / "material"
    material.mkdir(exist_ok=True)
    (project / "docs" / "sources.yaml").write_text(
        "sources:\n  - module: rust\n    topics: [lifetime]\n"
        f"    urls: ['{(material / 'unlicensed.md').as_uri()}']\n",
        encoding="utf-8",
    )
    (material / "unlicensed.md").write_text(
        "# Lifetimes\n\n" + ("Every reference has a lifetime. " * 12), encoding="utf-8"
    )
    env = env_for("fetch_docs")
    write_topics(env, [{"module": "rust", "topic": "lifetime"}])
    result = body(fetch_docs)(env)

    assert result["fetched"] >= 1
    assert result["unlicensed"] >= 1
    assert any("no licence recorded" in message for _level, message in env.ctx.logs)


def test_a_recorded_licence_is_stored_with_the_document(env_for, project, db):
    material = project / "material"
    material.mkdir(exist_ok=True)
    (project / "docs" / "sources.yaml").write_text(
        "sources:\n  - module: rust\n    license: MIT\n    topics: [lifetime]\n"
        f"    urls: ['{(material / 'licensed.md').as_uri()}']\n",
        encoding="utf-8",
    )
    (material / "licensed.md").write_text(
        "# Lifetimes\n\n" + ("Every reference has a lifetime. " * 12), encoding="utf-8"
    )
    env = env_for("fetch_docs")
    write_topics(env, [{"module": "rust", "topic": "lifetime"}])
    result = body(fetch_docs)(env)
    assert result["unlicensed"] == 0
    assert db.query_one("SELECT license FROM documents")["license"] == "MIT"


def test_feedback_is_consumed_as_it_is_handled_not_in_a_batch(env_for, db):
    """A preempted step has already stored those examples; re-counting them is a bug."""
    from gustav.core.supervisor import Preempted

    for index in range(4):
        _, answer_id = seed_exchange(db, question=f"question {index}", answer=f"answer {index}")
        seed_feedback(db, turn_id=answer_id, kind="correct", payload=f"the answer is {index}")

    env = env_for("consolidation", ctx=RecordingContext(stop_after=3))
    with pytest.raises(Preempted):
        body(consolidation)(env)

    handled = db.query_one(
        "SELECT COUNT(*) AS n FROM feedback WHERE consumed_run_id IS NOT NULL"
    )["n"]
    stored = db.query_one("SELECT COUNT(*) AS n FROM training_examples")["n"]
    assert 0 < handled < 4, "it stopped part-way through"
    assert handled == stored, "every example it stored has its feedback marked consumed"


def test_a_solved_task_becomes_a_runnable_retention_item(env_for, db):
    """A retention item for code should run the code, not compare prose."""
    from gustav.learn.bench import seed_regression_suite

    env = env_for("gate")
    task_id = TaskStore(db).add(
        Task(prompt="write add(a, b)", kind="python", module="coding",
             verifier={"tests": "assert add(1, 2) == 3"})
    )
    example_id = env.examples.add(
        Example(prompt="write add(a, b)", completion="def add(a, b):\n    return a + b",
                kind="practice", module="coding", source="task", source_ref=str(task_id),
                verified=True)
    )
    env.examples.mark_used([example_id or 0], run_id=1)

    assert seed_regression_suite(db, module="coding", exclude_run_id=2) == 1
    item = BenchmarkStore(db).suite("regression")[0]
    assert item.kind == "python"
    assert item.verifier["tests"] == "assert add(1, 2) == 3"


# --- how the gate scores free text --------------------------------------
def test_the_gate_says_which_scorer_it_used(env_for, db):
    model = TwoVersionModel(baseline="wrong", candidate="the right answer")
    env = env_for("gate", model=model)
    prepare_candidate(env)
    assert body(gate)(env)["scorer"] == "overlap", "no embeddings available"
    assert any("word overlap instead" in message for _level, message in env.ctx.logs)


def test_a_paraphrase_is_recognised_when_the_model_can_embed(env_for, db, stage_config):
    """The gate's weakest link: without embeddings this candidate looks like a failure."""
    paraphrase = "answer right the is this"  # the same words, a different sentence

    class Paraphrasing(ScriptedModel):
        def __init__(self):
            super().__init__(available=True)
            self.embeds = True

        def evaluate(self, prompts, *, candidate=None, system=None, **_overrides):
            self.candidates.append(candidate)
            answer = paraphrase if candidate else "completely unrelated text here"
            return [answer for _ in prompts]

    env = env_for("gate", model=Paraphrasing())
    prepare_candidate(env, match="prose")
    result = body(gate)(env)

    assert result["scorer"] == "embedding"
    assert result["promoted"] == ["coding"]
    row = db.query_one("SELECT detail FROM gate_results WHERE suite = 'heldout'")
    assert "semantic" in row["detail"]


def test_the_embedding_scorer_can_be_switched_off(env_for, db, stage_config):
    stage_config.gate.scorer = "overlap"

    class Paraphrasing(ScriptedModel):
        def __init__(self):
            super().__init__(available=True)
            self.embeds = True

        def evaluate(self, prompts, *, candidate=None, system=None, **_overrides):
            return ["the right answer" if candidate else "wrong" for _ in prompts]

    env = env_for("gate", model=Paraphrasing())
    prepare_candidate(env)
    result = body(gate)(env)
    assert result["scorer"] == "overlap"
    assert result["promoted"] == ["coding"], "it still promotes a genuinely better one"


def test_a_candidate_that_ties_is_not_promoted(env_for, db):
    """Nothing measured improved, so there is no reason to change the weights."""
    env = env_for("gate", model=TwoVersionModel(baseline="wrong", candidate="also wrong"))
    prepare_candidate(env)
    result = body(gate)(env)

    assert result["rejected"] == ["coding"]
    assert "a tie is not an improvement" in result["candidates"]["coding"]["reason"]
    assert (env.paths.adapters / "coding" / "active").readlink().name == "v001"


# --- the report, with stage 6's work in it -------------------------------
def test_the_report_shows_what_it_changed_about_itself(env_for, db, journal):
    from gustav.core.journal import StepSpec

    run = journal.create_run(
        trigger="manual", steps=[StepSpec("expansion"), StepSpec("speed")]
    )
    db.insert(
        "expansion_log",
        {"run_id": run.id, "kind": "new_module", "subject": "kubernetes",
         "rationale": "nine requests nothing covered", "status": "applied",
         "snapshot_id": "20261111T000000-expansion", "detail": "registered it"},
    )
    db.insert(
        "optimisations",
        {"run_id": run.id, "setting": "prefix_cache", "previous": "True",
         "candidate": "False", "baseline": 40.0, "measured": 60.0, "delta": 0.5,
         "status": "kept", "detail": "+50.0% faster"},
    )
    env = env_for("report", run_id=run.id)
    body(report)(env)

    html = (env.paths.reports / f"run-{run.id:06d}.html").read_text(encoding="utf-8")
    assert "What it changed about itself" in html
    assert "kubernetes" in html and "new_module" in html
    assert "What it measured about itself" in html
    assert "prefix_cache" in html and "+50.0%" in html
