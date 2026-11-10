from __future__ import annotations

import asyncio

import pytest

from gustav.config.schema import Config
from gustav.core.clock import FakeClock
from gustav.core.gpu_lease import GpuLease
from gustav.core.journal import Journal, RunStatus, StepSpec, StepStatus
from gustav.core.supervisor import Supervisor
from gustav.sleep.pipeline import (
    PLACEHOLDER,
    Pipeline,
    PipelineRunner,
    Step,
    default_pipeline,
)


def make_config(**sleep_kwargs) -> Config:
    return Config.model_validate(
        {
            "logging": {"console": False},
            "supervisor": {
                "max_attempts": 2,
                "retry_backoff_seconds": 0.01,
                "heartbeat_timeout_seconds": 1.0,
                "step_timeout_seconds": 20.0,
                "memory_limit_mb": 0,
                "kill_grace_seconds": 0.5,
            },
            "sleep": {"simulated_step_seconds": 0.02, **sleep_kwargs},
        }
    )


def small_pipeline(*names: str, gpu: tuple[str, ...] = ()) -> Pipeline:
    return Pipeline(
        name="test",
        steps=tuple(
            Step(StepSpec(n, requires_gpu=n in gpu), PLACEHOLDER) for n in names
        ),
    )


@pytest.fixture
def runner_factory(journal, events, paths):
    def build(config: Config, lease: GpuLease | None = None) -> PipelineRunner:
        return PipelineRunner(
            config=config,
            journal=journal,
            supervisor=Supervisor(poll_interval=0.02),
            events=events,
            lease=lease,
            clock=FakeClock(),
        )

    return build


def new_run(journal: Journal, paths, pipeline: Pipeline, **kwargs):
    run = journal.create_run(trigger="manual", steps=pipeline.specs, **kwargs)
    run_dir = paths.run_dir(run.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    journal.db.execute("UPDATE sleep_runs SET run_dir = ? WHERE id = ?", (str(run_dir), run.id))
    return journal.load_run(run.id)


async def test_full_run_completes(runner_factory, journal, paths):
    pipeline = small_pipeline("snapshot", "training", "gate")
    run = new_run(journal, paths, pipeline)
    outcome = await runner_factory(make_config()).run(run, pipeline)
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.completed == ["snapshot", "training", "gate"]
    reloaded = journal.load_run(run.id)
    assert all(s.status is StepStatus.DONE for s in reloaded.steps)
    assert reloaded.step("training").result["units"] == 4


async def test_each_step_leaves_its_artefacts(runner_factory, journal, paths):
    pipeline = small_pipeline("snapshot")
    run = new_run(journal, paths, pipeline)
    await runner_factory(make_config()).run(run, pipeline)
    units = sorted(p.name for p in (paths.run_dir(run.id) / "snapshot").iterdir())
    assert units == ["unit-00.done", "unit-01.done", "unit-02.done", "unit-03.done"]


async def test_already_done_steps_are_not_run_again(runner_factory, journal, paths):
    pipeline = small_pipeline("snapshot", "training")
    run = new_run(journal, paths, pipeline)
    journal.finish_step(run.id, "snapshot", StepStatus.DONE, result={"from": "earlier"})
    outcome = await runner_factory(make_config()).run(journal.load_run(run.id), pipeline)
    assert outcome.completed == ["snapshot", "training"]
    assert journal.load_run(run.id).step("snapshot").result == {"from": "earlier"}
    assert not (paths.run_dir(run.id) / "snapshot").exists()


async def test_a_step_resumes_from_its_checkpoint(runner_factory, journal, paths):
    pipeline = small_pipeline("training")
    run = new_run(journal, paths, pipeline)
    journal.set_checkpoint(run.id, "training", {"unit": 3})
    await runner_factory(make_config()).run(journal.load_run(run.id), pipeline)
    # It picked up at unit 3 instead of redoing the first three.
    assert journal.load_run(run.id).step("training").result["resumed_from"] == 3
    units = sorted(p.name for p in (paths.run_dir(run.id) / "training").iterdir())
    assert units == ["unit-03.done"]


async def test_transient_failure_is_retried(runner_factory, journal, paths, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "training=fail:once")
    pipeline = small_pipeline("training")
    run = new_run(journal, paths, pipeline)
    outcome = await runner_factory(make_config()).run(run, pipeline)
    assert outcome.status is RunStatus.COMPLETED
    step = journal.load_run(run.id).step("training")
    assert step.status is StepStatus.DONE and step.attempts == 2


async def test_persistent_failure_is_given_up_on_and_the_night_continues(
    runner_factory, journal, paths, monkeypatch
):
    monkeypatch.setenv("GUSTAV_FAULT", "study=fail")
    pipeline = small_pipeline("study", "training")
    run = new_run(journal, paths, pipeline)
    outcome = await runner_factory(make_config()).run(run, pipeline)
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.failed == ["study"] and outcome.completed == ["training"]
    step = journal.load_run(run.id).step("study")
    assert step.status is StepStatus.FAILED
    assert "gave up after 2 attempts" in step.error
    assert step.attempts == 3  # two real attempts plus the one that reports the give-up


async def test_a_critical_step_failing_stops_the_run(runner_factory, journal, paths, monkeypatch):
    monkeypatch.setenv("GUSTAV_FAULT", "snapshot=fail")
    pipeline = Pipeline(
        name="test",
        steps=(
            Step(StepSpec("snapshot", critical=True), PLACEHOLDER),
            Step(StepSpec("training"), PLACEHOLDER),
        ),
    )
    run = new_run(journal, paths, pipeline)
    outcome = await runner_factory(make_config()).run(run, pipeline)
    assert outcome.status is RunStatus.FAILED
    assert journal.load_run(run.id).step("training").status is StepStatus.PENDING
    assert journal.load_run(run.id).status is RunStatus.FAILED


async def test_a_hanging_step_is_killed_and_retried(
    runner_factory, journal, paths, monkeypatch
):
    monkeypatch.setenv("GUSTAV_FAULT", "training=hang:once")
    pipeline = small_pipeline("training")
    run = new_run(journal, paths, pipeline)
    outcome = await runner_factory(make_config()).run(run, pipeline)
    assert outcome.status is RunStatus.COMPLETED
    assert journal.load_run(run.id).step("training").attempts == 2


async def test_a_crashing_step_does_not_crash_the_runner(
    runner_factory, journal, paths, monkeypatch
):
    monkeypatch.setenv("GUSTAV_FAULT", "training=crash:once")
    pipeline = small_pipeline("training", "gate")
    run = new_run(journal, paths, pipeline)
    outcome = await runner_factory(make_config()).run(run, pipeline)
    assert outcome.status is RunStatus.COMPLETED
    assert outcome.completed == ["training", "gate"]


async def test_preemption_pauses_the_run_with_a_checkpoint(
    runner_factory, journal, paths, monkeypatch
):
    monkeypatch.setenv("GUSTAV_FAULT", "training=pause")
    pipeline = small_pipeline("snapshot", "training", "gate")
    run = new_run(journal, paths, pipeline)
    stop = asyncio.Event()

    async def interrupt():
        await asyncio.sleep(0.6)
        stop.set()

    asyncio.create_task(interrupt())
    outcome = await runner_factory(make_config()).run(run, pipeline, stop_event=stop)

    assert outcome.status is RunStatus.PAUSED
    assert outcome.preempted_at == "training"
    reloaded = journal.load_run(run.id)
    assert reloaded.status is RunStatus.PAUSED
    assert reloaded.step("snapshot").status is StepStatus.DONE
    assert reloaded.step("training").status is StepStatus.PREEMPTED
    # Progress up to the pause point was recorded, so resuming redoes almost nothing.
    assert reloaded.step("training").checkpoint["unit"] == 2
    assert reloaded.step("gate").status is StepStatus.PENDING


async def test_a_paused_run_resumes_where_it_stopped(
    runner_factory, journal, paths, monkeypatch
):
    monkeypatch.setenv("GUSTAV_FAULT", "training=pause")
    pipeline = small_pipeline("snapshot", "training", "gate")
    run = new_run(journal, paths, pipeline)
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(0.6, stop.set)
    await runner_factory(make_config()).run(run, pipeline, stop_event=stop)

    # Second night: no fault, resume from the journal.
    monkeypatch.delenv("GUSTAV_FAULT")
    journal.reset_step(run.id, "training")
    outcome = await runner_factory(make_config()).run(
        journal.load_run(run.id), pipeline
    )
    assert outcome.status is RunStatus.COMPLETED
    step = journal.load_run(run.id).step("training")
    assert step.status is StepStatus.DONE
    assert step.result["resumed_from"] == 2


async def test_stop_before_the_first_step_pauses_immediately(runner_factory, journal, paths):
    pipeline = small_pipeline("snapshot", "training")
    run = new_run(journal, paths, pipeline)
    stop = asyncio.Event()
    stop.set()
    outcome = await runner_factory(make_config()).run(run, pipeline, stop_event=stop)
    assert outcome.status is RunStatus.PAUSED
    assert journal.load_run(run.id).step("snapshot").status is StepStatus.PENDING


async def test_deadline_pauses_the_run(runner_factory, journal, paths):
    pipeline = small_pipeline("snapshot", "training")
    run = new_run(journal, paths, pipeline)
    runner = runner_factory(make_config())
    outcome = await runner.run(run, pipeline, deadline=runner.clock.monotonic() - 1)
    assert outcome.status is RunStatus.PAUSED
    assert "maximum duration" in outcome.reason


async def test_gpu_steps_take_and_return_the_lease(runner_factory, journal, paths):
    lease = GpuLease()
    holders: list[str | None] = []

    pipeline = small_pipeline("snapshot", "training", gpu=("training",))
    run = new_run(journal, paths, pipeline)
    runner = runner_factory(make_config(), lease)

    original = runner._on_child_message

    def spy(run_id, step, msg):
        if msg.get("type") == "heartbeat":
            holders.append(lease.holder)
        original(run_id, step, msg)

    runner._on_child_message = spy
    await runner.run(run, pipeline)

    assert lease.holder is None  # returned at the end of the step
    assert "sleep:training" in holders
    assert None in holders or "snapshot" not in holders  # snapshot never claimed it


async def test_the_lease_is_returned_even_when_a_step_fails(
    runner_factory, journal, paths, monkeypatch
):
    monkeypatch.setenv("GUSTAV_FAULT", "training=fail")
    lease = GpuLease()
    pipeline = small_pipeline("training", gpu=("training",))
    run = new_run(journal, paths, pipeline)
    await runner_factory(make_config(), lease).run(run, pipeline)
    assert lease.holder is None


async def test_disabled_steps_are_not_in_the_pipeline():
    config = make_config(disabled_steps=["dreaming", "speed"])
    names = [s.spec.name for s in default_pipeline(config).steps]
    assert "dreaming" not in names and "speed" not in names
    assert "training" in names


async def test_dry_run_writes_no_artefacts(runner_factory, journal, paths):
    pipeline = small_pipeline("training")
    run = new_run(journal, paths, pipeline, dry_run=True)
    outcome = await runner_factory(make_config()).run(run, pipeline)
    assert outcome.status is RunStatus.COMPLETED
    assert not any((paths.run_dir(run.id) / "training").iterdir())
    assert journal.load_run(run.id).step("training").result["dry_run"] is True


async def test_events_tell_the_story_of_the_night(runner_factory, journal, paths, events):
    pipeline = small_pipeline("snapshot", "training")
    run = new_run(journal, paths, pipeline)
    await runner_factory(make_config()).run(run, pipeline)
    kinds = [e["kind"] for e in events.recent(limit=50)]
    assert kinds.count("sleep.step.start") == 2
    assert kinds.count("sleep.step.done") == 2
    assert "sleep.completed" in kinds


def test_the_default_pipeline_matches_the_spec():
    names = [s.spec.name for s in default_pipeline(make_config()).steps]
    assert names[0] == "snapshot", "nothing may run before a verified backup"
    assert names.index("gate") < names.index("expansion")
    assert names.index("training") < names.index("gate")
    assert names[-1] == "report"
    # Everything that can touch weights must have a crash policy that is not 'resume'.
    specs = {s.spec.name: s.spec for s in default_pipeline(make_config()).steps}
    assert specs["expansion"].on_crash == "rollback"
    assert specs["snapshot"].critical and specs["gate"].critical
