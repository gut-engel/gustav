"""The stores the night reads and writes."""

from __future__ import annotations

import pytest

from gustav.learn.examples import Example, ExampleStore
from gustav.learn.facts import Fact, FactStore
from gustav.learn.fingerprint import bucket, fingerprint, normalise
from gustav.learn.skills import SkillStore, smoothed
from gustav.learn.tasks import Task, TaskStore


# --- fingerprints -------------------------------------------------------
def test_normalisation_ignores_differences_that_are_not_differences():
    assert normalise("  The  Editor is Helix. ") == "the editor is helix"
    assert fingerprint("a", "b") == fingerprint("A", " b ")
    assert fingerprint("a", "b") != fingerprint("b", "a")


def test_bucket_is_stable_and_spread():
    values = [bucket(fingerprint(f"prompt {i}")) for i in range(200)]
    assert all(0.0 <= v <= 1.0 for v in values)
    # A stable hash-derived split, not a shuffle: same input, same side of the line.
    assert bucket(fingerprint("prompt 7")) == values[7]
    assert 0.08 < sum(v < 0.15 for v in values) / len(values) < 0.25


# --- examples -----------------------------------------------------------
def test_the_same_example_twice_is_stored_once(db):
    store = ExampleStore(db)
    first = store.add(Example(prompt="q", completion="a", kind="study"))
    second = store.add(Example(prompt=" Q ", completion="A", kind="study"))
    assert first is not None
    assert second is None, "normalised duplicates must not become two examples"


def test_corrections_are_never_held_out(db):
    store = ExampleStore(db, heldout_fraction=1.0)
    store.add(Example(prompt="q", completion="a", kind="correction", verified=True))
    store.add(Example(prompt="q2", completion="a2", kind="study"))
    assert [r["kind"] for r in store.for_training()] == ["correction"]
    assert [r["kind"] for r in store.heldout()] == ["study"]


def test_heldout_examples_are_unreachable_from_training(db):
    store = ExampleStore(db, heldout_fraction=0.5)
    store.add_many(
        Example(prompt=f"q{i}", completion=f"a{i}", kind="study") for i in range(40)
    )
    heldout = {r["id"] for r in store.heldout(limit=100)}
    training = {r["id"] for r in store.for_training(limit=100)}
    assert heldout and training
    assert not heldout & training


def test_the_split_does_not_move_between_runs(db):
    first = ExampleStore(db, heldout_fraction=0.5)
    example = Example(prompt="what is a lifetime", completion="a scope", kind="study")
    side = first.split_for(example)
    # A different store object, a different night: the same example, the same side.
    assert ExampleStore(db, heldout_fraction=0.5).split_for(example) == side


def test_weights_default_by_kind_and_order_the_mix(db):
    store = ExampleStore(db)
    store.add(Example(prompt="q1", completion="a", kind="study"))
    store.add(Example(prompt="q2", completion="a", kind="correction"))
    kinds = [r["kind"] for r in store.for_training()]
    assert kinds[0] == "correction", "a correction outranks a study example"


def test_replay_only_offers_what_was_already_trained_on(db):
    store = ExampleStore(db, heldout_fraction=0.0)
    ids = store.add_many(
        Example(prompt=f"q{i}", completion="a", kind="study") for i in range(10)
    ).added
    assert store.replay(limit=5) == []
    store.mark_used(ids[:6], run_id=1)
    replayed = store.replay(limit=4)
    assert len(replayed) == 4
    assert all(r["status"] == "used" for r in replayed)
    # And the used ones are out of the fresh pool.
    assert len(store.for_training()) == 4


def test_forgetting_a_turn_withdraws_its_examples(db):
    store = ExampleStore(db)
    store.add(Example(prompt="q", completion="a", kind="fact", source="turn", source_ref="7"))
    assert store.drop_for_turns([7]) == 1
    assert store.for_training() == []


# --- facts --------------------------------------------------------------
def test_a_newer_fact_supersedes_the_old_one_without_deleting_it(db):
    facts = FactStore(db)
    facts.add(Fact(text="The user's editor is vim.", subject="editor"))
    change = facts.add(Fact(text="The user's editor is helix.", subject="editor"))
    assert change is not None and len(change.superseded) == 1
    assert [f["text"] for f in facts.active()] == ["The user's editor is helix."]
    assert len(facts.history("editor")) == 2, "the old statement is still readable"


def test_facts_without_a_subject_coexist(db):
    facts = FactStore(db)
    facts.add(Fact(text="The user prefers short answers.", subject=None))
    facts.add(Fact(text="The user prefers metric units.", subject=None))
    assert len(facts.active()) == 2


def test_repeating_a_fact_raises_confidence_rather_than_duplicating(db):
    facts = FactStore(db)
    facts.add(Fact(text="The user lives in Berlin.", subject="location", confidence=0.6))
    assert facts.add(Fact(text="The user lives in Berlin.", subject="location")) is None
    assert facts.active()[0]["confidence"] == pytest.approx(0.65)


def test_forgetting_a_turn_retracts_the_facts_it_produced(db):
    facts = FactStore(db)
    facts.add(Fact(text="The user's name is Simon.", subject="name", source_turn_id=3))
    assert facts.retract_for_turns([3]) == 1
    assert facts.active() == []


# --- skills -------------------------------------------------------------
def test_one_lucky_pass_is_not_a_skill():
    assert smoothed(1, 1) < 0.7
    assert smoothed(20, 20) > 0.9
    assert smoothed(0, 0) == 0.5


def test_weakness_ranks_on_demand_as_well_as_score(db):
    skills = SkillStore(db)
    for _ in range(5):
        skills.record("rust", "lifetime", passed=False)
        skills.record("coding", "regex", passed=False)
    for _ in range(4):
        skills.note_demand("rust", "lifetime")
    weakest = skills.weakest(limit=2)
    assert str(weakest[0].topic) == "lifetime", "equally bad, but one is actually asked about"


def test_a_measured_strength_is_not_offered_for_study(db):
    skills = SkillStore(db)
    for _ in range(10):
        skills.record("coding", "regex", passed=True)
    assert skills.weakest(threshold=0.6) == []


# --- tasks --------------------------------------------------------------
def test_attempts_accumulate_and_a_pass_solves_the_task(db):
    tasks = TaskStore(db)
    task_id = tasks.add(
        Task(prompt="add two numbers", kind="python", verifier={"tests": "assert add(1,2)==3"})
    )
    assert task_id is not None
    tasks.record_attempt(task_id, run_id=1, step="practice", answer="x", passed=False,
                         detail="AssertionError")
    assert tasks.counts() == {"open": 1}
    tasks.record_attempt(task_id, run_id=1, step="dreaming", answer="y", passed=True)
    assert tasks.counts() == {"solved": 1}
    assert [a["attempt"] for a in tasks.attempts(task_id)] == [1, 2]


def test_an_unknown_task_kind_is_refused(db):
    with pytest.raises(ValueError, match="unknown task kind"):
        TaskStore(db).add(Task(prompt="x", kind="telepathy"))


def test_failed_tasks_are_offered_before_untried_ones(db):
    tasks = TaskStore(db)
    fresh = tasks.add(Task(prompt="fresh", kind="python", verifier={"tests": "assert 1"}))
    tried = tasks.add(Task(prompt="tried", kind="python", verifier={"tests": "assert 2"}))
    tasks.record_attempt(tried or 0, run_id=1, step="practice", answer="", passed=False)
    assert [t.prompt for t in tasks.open()] == ["tried", "fresh"]
    assert fresh is not None


def test_restating_an_old_fact_supersedes_the_newer_one(db):
    """'vim, no wait, helix again' has to end with one active editor, not two."""
    facts = FactStore(db)
    facts.add(Fact(text="The user's editor is helix.", subject="editor"))
    facts.add(Fact(text="The user's editor is vim.", subject="editor"))
    change = facts.add(Fact(text="The user's editor is helix.", subject="editor"))

    assert change is not None and not change.is_new, "the old row is revived, not duplicated"
    assert [f["text"] for f in facts.active()] == ["The user's editor is helix."]
    assert len(facts.history("editor")) == 2
