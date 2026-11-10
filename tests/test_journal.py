from __future__ import annotations

import pytest

from gustav.core.journal import (
    CrashPolicy,
    Journal,
    RunStatus,
    StepSpec,
    StepStatus,
)

STEPS = [
    StepSpec("snapshot", CrashPolicy.RESTART, critical=True),
    StepSpec("training", CrashPolicy.RESUME),
    StepSpec("gate", CrashPolicy.RESTART),
]


def test_create_run_records_every_step_up_front(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    assert [s.name for s in run.steps] == ["snapshot", "training", "gate"]
    assert all(s.status is StepStatus.PENDING for s in run.steps)
    assert run.status is RunStatus.RUNNING
    assert run.step("snapshot").on_crash is CrashPolicy.RESTART


def test_next_step_walks_the_pipeline(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    assert run.next_step.name == "snapshot"
    journal.finish_step(run.id, "snapshot", StepStatus.DONE)
    assert journal.load_run(run.id).next_step.name == "training"
    journal.finish_step(run.id, "training", StepStatus.SKIPPED)
    assert journal.load_run(run.id).next_step.name == "gate"
    journal.finish_step(run.id, "gate", StepStatus.DONE)
    assert journal.load_run(run.id).next_step is None


def test_failed_step_is_not_terminal_for_resumption(journal: Journal):
    """A failed step is retried on the next pass; only done/skipped are final."""
    run = journal.create_run(trigger="manual", steps=STEPS)
    journal.finish_step(run.id, "snapshot", StepStatus.FAILED, error="disk full")
    assert journal.load_run(run.id).next_step.name == "snapshot"


def test_attempts_increment(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    assert journal.begin_step(run.id, "training") == 1
    assert journal.begin_step(run.id, "training") == 2


def test_checkpoints_survive_reload(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    journal.set_checkpoint(run.id, "training", {"unit": 3, "loss": 1.5})
    assert journal.load_run(run.id).step("training").checkpoint == {"unit": 3, "loss": 1.5}


def test_finishing_a_step_can_clear_its_checkpoint(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    journal.set_checkpoint(run.id, "training", {"unit": 3})
    journal.finish_step(run.id, "training", StepStatus.DONE, result={"ok": True},
                        clear_checkpoint=True)
    step = journal.load_run(run.id).step("training")
    assert step.checkpoint is None and step.result == {"ok": True}


def test_reset_step_keeps_or_clears_the_checkpoint(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    journal.set_checkpoint(run.id, "training", {"unit": 2})
    journal.reset_step(run.id, "training")
    assert journal.load_run(run.id).step("training").checkpoint == {"unit": 2}
    journal.reset_step(run.id, "training", clear_checkpoint=True)
    assert journal.load_run(run.id).step("training").checkpoint is None


def test_open_runs_and_terminal_runs(journal: Journal):
    first = journal.create_run(trigger="window", steps=STEPS)
    journal.set_run_status(first.id, RunStatus.COMPLETED)
    second = journal.create_run(trigger="manual", steps=STEPS)
    assert [r.id for r in journal.open_runs()] == [second.id]
    journal.set_run_status(second.id, RunStatus.PAUSED, reason="user message")
    assert [r.id for r in journal.open_runs()] == [second.id]  # paused is still open


def test_finishing_a_run_stamps_finished_at(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    assert journal.load_run(run.id).finished_at is None
    journal.set_run_status(run.id, RunStatus.COMPLETED, reason="done")
    reloaded = journal.load_run(run.id)
    assert reloaded.finished_at is not None and reloaded.reason == "done"


def test_unknown_step_is_an_error(journal: Journal):
    run = journal.create_run(trigger="manual", steps=STEPS)
    with pytest.raises(KeyError):
        journal.begin_step(run.id, "no-such-step")


def test_unknown_run_is_an_error(journal: Journal):
    with pytest.raises(KeyError):
        journal.load_run(4242)


def test_pending_work_accumulates_and_is_consumed(journal: Journal):
    assert journal.pending_work_count() == 0
    journal.add_pending_work("turn", ref="1")
    journal.add_pending_work("correction", weight=10, ref="2")
    assert journal.pending_work_count() == 11
    run = journal.create_run(trigger="manual", steps=STEPS)
    assert journal.consume_pending_work(run.id) == 2
    assert journal.pending_work_count() == 0


def test_two_runs_cannot_share_a_step_row(journal: Journal):
    a = journal.create_run(trigger="manual", steps=STEPS)
    b = journal.create_run(trigger="manual", steps=STEPS)
    journal.finish_step(a.id, "training", StepStatus.DONE)
    assert journal.load_run(b.id).step("training").status is StepStatus.PENDING
