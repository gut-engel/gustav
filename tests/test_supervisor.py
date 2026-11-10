from __future__ import annotations

import asyncio

import pytest

from gustav.core.supervisor import ChildSpec, Outcome, Supervisor

CHILDREN = "gustav.testing.children"


def spec(name: str, **kwargs) -> ChildSpec:
    defaults = dict(
        name=name,
        target=f"{CHILDREN}:{name}",
        timeout_seconds=15.0,
        heartbeat_timeout_seconds=2.0,
        memory_limit_mb=0,
        kill_grace_seconds=0.5,
    )
    defaults.update(kwargs)
    return ChildSpec(**defaults)


@pytest.fixture
def sup() -> Supervisor:
    return Supervisor(poll_interval=0.02)


async def test_successful_child_returns_its_payload(sup: Supervisor):
    result = await sup.run(spec("ok_child", kwargs={"beats": 3, "value": 7}))
    assert result.ok and result.outcome is Outcome.OK
    assert result.payload["value"] == 7
    assert result.heartbeats >= 3
    assert result.exit_code == 0


async def test_child_runs_in_its_own_process(sup: Supervisor):
    import os

    result = await sup.run(spec("ok_child"))
    assert result.payload["pid"] != os.getpid()


async def test_exception_in_child_is_reported_not_raised(sup: Supervisor):
    result = await sup.run(spec("failing_child", kwargs={"message": "kaboom"}))
    assert result.outcome is Outcome.FAILED
    assert "kaboom" in result.error
    assert "Traceback" in result.error  # the child's traceback comes back too
    assert result.outcome.retryable


async def test_hanging_child_is_killed_by_the_watchdog(sup: Supervisor):
    result = await sup.run(spec("hanging_child", heartbeat_timeout_seconds=0.5))
    assert result.outcome is Outcome.HANG
    assert "no heartbeat" in result.error
    assert result.duration_seconds < 5


async def test_wall_clock_timeout_fires_even_while_heartbeating(sup: Supervisor):
    result = await sup.run(
        spec("busy_child", kwargs={"seconds": 30}, timeout_seconds=0.6,
             heartbeat_timeout_seconds=10)
    )
    assert result.outcome is Outcome.TIMEOUT
    assert result.duration_seconds < 5


async def test_segfault_style_death_is_classified_not_crashing_the_parent(sup: Supervisor):
    result = await sup.run(spec("crashing_child"))
    assert result.outcome in (Outcome.OOM, Outcome.CRASHED)
    assert "signal" in (result.error or "")


async def test_memory_limit_contains_a_runaway_child(sup: Supervisor):
    result = await sup.run(
        spec("oom_child", memory_limit_mb=256, heartbeat_timeout_seconds=10,
             timeout_seconds=30)
    )
    assert result.outcome is Outcome.OOM


async def test_parent_survives_every_kind_of_child_death(sup: Supervisor):
    """The point of child processes: chat stays alive whatever happens in there."""
    for target, kwargs in [
        ("failing_child", {}),
        ("crashing_child", {}),
        ("hanging_child", {}),
    ]:
        await sup.run(spec(target, kwargs=kwargs, heartbeat_timeout_seconds=0.5,
                           timeout_seconds=5))
    # Still able to run a healthy child afterwards.
    assert (await sup.run(spec("ok_child"))).ok


async def test_checkpoints_arrive_as_they_happen(sup: Supervisor):
    seen: list[dict] = []
    result = await sup.run(
        spec("checkpointing_child", kwargs={"units": 4}),
        on_message=lambda m: seen.append(m) if m["type"] == "checkpoint" else None,
    )
    assert result.ok
    assert [m["data"]["unit"] for m in seen] == [1, 2, 3, 4]
    assert result.checkpoint == {"unit": 4}


async def test_stop_request_makes_a_cooperative_child_checkpoint_and_exit(sup: Supervisor):
    stop = asyncio.Event()
    checkpoints: list[dict] = []

    async def stop_soon():
        await asyncio.sleep(0.3)
        stop.set()

    asyncio.create_task(stop_soon())
    result = await sup.run(
        spec("stoppable_child", timeout_seconds=10),
        stop_event=stop,
        on_message=lambda m: checkpoints.append(m["data"]) if m["type"] == "checkpoint" else None,
    )
    assert result.outcome is Outcome.PREEMPTED
    assert checkpoints and checkpoints[-1].get("final") is True
    assert result.exit_code == 17


async def test_child_that_ignores_the_stop_request_is_killed(sup: Supervisor):
    """A sleep step must give the GPU back; cooperation is preferred, not required."""
    stop = asyncio.Event()
    stop.set()
    result = await sup.run(
        spec("ignoring_child", timeout_seconds=30, heartbeat_timeout_seconds=30,
             stop_grace_seconds=0.5),
        stop_event=stop,
    )
    assert result.outcome is Outcome.PREEMPTED
    assert "did not stop" in result.error
    assert result.duration_seconds < 5


async def test_two_children_can_run_concurrently(sup: Supervisor):
    results = await asyncio.gather(
        sup.run(spec("ok_child", kwargs={"value": 1})),
        sup.run(spec("ok_child", kwargs={"value": 2})),
    )
    assert sorted(r.payload["value"] for r in results) == [1, 2]
    assert results[0].payload["pid"] != results[1].payload["pid"]


async def test_bad_target_is_reported_as_a_failure(sup: Supervisor):
    result = await sup.run(spec("nope", target="gustav.testing.children:does_not_exist"))
    assert result.outcome is Outcome.FAILED
    assert "AttributeError" in result.error


async def test_unimportable_module_is_reported(sup: Supervisor):
    result = await sup.run(spec("nope", target="no.such.module:fn"))
    assert result.outcome is Outcome.FAILED
    assert "ModuleNotFoundError" in result.error


async def test_malformed_target_is_reported(sup: Supervisor):
    result = await sup.run(spec("nope", target="gustav.testing.children"))
    assert result.outcome is Outcome.FAILED
    assert "module:function" in result.error
