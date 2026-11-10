from __future__ import annotations

import pytest

from gustav.core.events import Events
from gustav.core.journal import CrashPolicy, Journal, RunStatus, StepSpec, StepStatus
from gustav.core.recovery import Recovery

STEPS = [
    StepSpec("snapshot", CrashPolicy.RESTART, critical=True),
    StepSpec("study", CrashPolicy.RESUME),
    StepSpec("training", CrashPolicy.RESUME),
    StepSpec("expansion", CrashPolicy.ROLLBACK),
    StepSpec("report", CrashPolicy.SKIP),
]


@pytest.fixture
def recovery_factory(journal: Journal, events: Events, paths):
    def build(rollbacks=None, health_checks=None) -> Recovery:
        return Recovery(
            journal=journal,
            events=events,
            paths=paths,
            rollbacks=rollbacks,
            health_checks=health_checks,
        )

    return build


def interrupted_run(journal: Journal, paths, running_step: str, *, done: list[str] = ()):
    run = journal.create_run(trigger="window", steps=STEPS, run_dir=paths.run_dir(1))
    for name in done:
        journal.finish_step(run.id, name, StepStatus.DONE)
    journal.begin_step(run.id, running_step)
    journal.set_checkpoint(run.id, running_step, {"unit": 2})
    return journal.load_run(run.id)


def test_clean_startup_is_reported_as_clean(recovery_factory):
    report = recovery_factory().run()
    assert report.clean and report.summary() == "clean startup"


def test_a_running_run_is_marked_interrupted(recovery_factory, journal, paths):
    run = interrupted_run(journal, paths, "study", done=["snapshot"])
    report = recovery_factory().run()
    assert report.interrupted_runs == [run.id]
    assert report.resumable_runs == [run.id]
    assert journal.load_run(run.id).status is RunStatus.INTERRUPTED


def test_resume_policy_keeps_the_checkpoint(recovery_factory, journal, paths):
    run = interrupted_run(journal, paths, "training", done=["snapshot", "study"])
    recovery_factory().run()
    step = journal.load_run(run.id).step("training")
    assert step.status is StepStatus.PENDING
    assert step.checkpoint == {"unit": 2}  # it will pick up at unit 2
    assert journal.load_run(run.id).next_step.name == "training"


def test_restart_policy_discards_the_checkpoint_and_the_work_dir(
    recovery_factory, journal, paths
):
    run = interrupted_run(journal, paths, "snapshot")
    work = paths.run_dir(run.id) / "snapshot"
    work.mkdir(parents=True, exist_ok=True)
    (work / "half-written.tmp").write_text("partial", encoding="utf-8")

    report = recovery_factory().run()
    step = journal.load_run(run.id).step("snapshot")
    assert step.status is StepStatus.PENDING
    assert step.checkpoint is None
    assert not work.exists()
    assert f"{run.id}:snapshot" in report.restarted_steps


def test_skip_policy_gives_up_on_the_step(recovery_factory, journal, paths):
    run = interrupted_run(journal, paths, "report", done=["snapshot", "study", "training",
                                                         "expansion"])
    report = recovery_factory().run()
    assert journal.load_run(run.id).step("report").status is StepStatus.SKIPPED
    assert f"{run.id}:report" in report.skipped_steps
    # Nothing left to do, so the run is closed rather than left dangling.
    assert journal.load_run(run.id).status is RunStatus.COMPLETED


def test_rollback_policy_calls_the_handler_and_fails_the_run(
    recovery_factory, journal, paths
):
    run = interrupted_run(journal, paths, "expansion", done=["snapshot", "study", "training"])
    called: list[tuple[int, str]] = []
    report = recovery_factory(rollbacks={"expansion": lambda r, s: called.append((r, s))}).run()
    assert called == [(run.id, "expansion")]
    assert journal.load_run(run.id).status is RunStatus.FAILED
    assert report.failed_runs == [run.id]
    assert journal.load_run(run.id).step("expansion").status is StepStatus.FAILED


def test_a_missing_rollback_handler_is_reported_not_ignored(
    recovery_factory, journal, paths
):
    interrupted_run(journal, paths, "expansion")
    report = recovery_factory().run()
    assert any("no handler is registered" in m for m in report.messages)
    assert report.rolled_back_steps


def test_a_failing_rollback_handler_does_not_block_startup(
    recovery_factory, journal, paths
):
    interrupted_run(journal, paths, "expansion")

    def boom(run_id, step):
        raise RuntimeError("restore failed")

    report = recovery_factory(rollbacks={"expansion": boom}).run()
    assert any("restore failed" in m for m in report.messages)


def test_a_preempted_step_is_queued_for_resumption(recovery_factory, journal, paths):
    run = journal.create_run(trigger="manual", steps=STEPS)
    journal.finish_step(run.id, "snapshot", StepStatus.DONE)
    journal.finish_step(run.id, "study", StepStatus.PREEMPTED, checkpoint={"unit": 1})
    journal.set_run_status(run.id, RunStatus.PAUSED, reason="user message")

    report = recovery_factory().run()
    step = journal.load_run(run.id).step("study")
    assert step.status is StepStatus.PENDING and step.checkpoint == {"unit": 1}
    assert report.resumable_runs == [run.id]
    # A paused run was not interrupted: it stopped on purpose.
    assert report.interrupted_runs == []


def test_staging_directories_are_cleaned(recovery_factory, paths):
    staged = paths.tmp / "adapter-v003-partial"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "model.safetensors.part").write_text("half", encoding="utf-8")
    loose = paths.tmp / "stray.tmp"
    loose.write_text("x", encoding="utf-8")

    report = recovery_factory().run()
    assert sorted(report.cleaned_staging) == ["adapter-v003-partial", "stray.tmp"]
    assert not staged.exists() and not loose.exists()
    assert not report.clean


def test_a_failing_health_check_puts_us_in_safe_mode(recovery_factory):
    report = recovery_factory(
        health_checks=[lambda: (False, "active adapter v007 will not load")]
    ).run()
    assert report.safe_mode
    assert "will not load" in report.messages[0]
    assert "SAFE MODE" in report.summary()


def test_a_raising_health_check_is_treated_as_a_failure(recovery_factory):
    def boom():
        raise OSError("no such file")

    report = recovery_factory(health_checks=[boom]).run()
    assert report.safe_mode
    assert "OSError" in report.messages[0]


def test_recovery_is_idempotent(recovery_factory, journal, paths):
    run = interrupted_run(journal, paths, "training", done=["snapshot", "study"])
    first = recovery_factory().run()
    second = recovery_factory().run()
    assert first.resumable_runs == [run.id]
    assert second.interrupted_runs == []  # already marked
    assert second.resumable_runs == [run.id]
    assert journal.load_run(run.id).step("training").checkpoint == {"unit": 2}


def test_recovery_writes_an_event(recovery_factory, journal, paths, events):
    interrupted_run(journal, paths, "training")
    recovery_factory().run()
    kinds = [e["kind"] for e in events.recent()]
    assert "daemon.recovery" in kinds
