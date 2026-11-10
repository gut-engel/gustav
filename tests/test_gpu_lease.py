from __future__ import annotations

import asyncio

import pytest

from gustav.core.gpu_lease import GpuBusy, GpuLease


async def test_acquire_and_release():
    lease = GpuLease()
    await lease.acquire("inference")
    assert lease.holder == "inference"
    await lease.release("inference")
    assert lease.holder is None


async def test_same_holder_is_reentrant():
    lease = GpuLease()
    await lease.acquire("inference")
    await asyncio.wait_for(lease.acquire("inference"), timeout=1)
    assert lease.holder == "inference"


async def test_inference_is_evicted_for_training():
    """The real single-GPU case: training takes the card, inference unloads first."""
    lease = GpuLease()
    unloaded = asyncio.Event()

    async def evict():
        unloaded.set()
        await lease.release("inference")

    await lease.acquire("inference", evictor=evict)
    await lease.acquire("sleep:training", reason="QLoRA", timeout=5)
    assert unloaded.is_set()
    assert lease.holder == "sleep:training"


async def test_a_holder_without_an_evictor_must_cooperate():
    """A training child has no evictor: the lease asks, and waits for it to release."""
    lease = GpuLease()
    await lease.acquire("sleep:training")
    waiting = asyncio.create_task(lease.acquire("inference", timeout=5))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    assert lease.return_requested  # the step can see it has to wind up

    await lease.release("sleep:training")
    await asyncio.wait_for(waiting, timeout=2)
    assert lease.holder == "inference"


async def test_waiting_times_out_rather_than_hanging_forever():
    lease = GpuLease()
    await lease.acquire("sleep:training")
    with pytest.raises(GpuBusy) as exc:
        await lease.acquire("inference", timeout=0.2)
    assert exc.value.holder == "sleep:training"


async def test_request_return_sets_the_flag_and_evicts():
    lease = GpuLease()
    evicted = asyncio.Event()
    await lease.acquire("inference", evictor=lambda: evicted.set())
    assert await lease.request_return("user message")
    assert evicted.is_set() and lease.return_requested


async def test_request_return_on_a_free_lease_is_false():
    assert not await GpuLease().request_return("nobody there")


async def test_release_by_the_wrong_holder_is_ignored():
    lease = GpuLease()
    await lease.acquire("inference")
    await lease.release("someone-else")
    assert lease.holder == "inference"


async def test_hold_context_manager_releases_on_error():
    lease = GpuLease()
    with pytest.raises(ValueError):
        async with lease.hold("sleep:training"):
            raise ValueError("step blew up")
    assert lease.holder is None


async def test_queue_of_waiters_is_served():
    lease = GpuLease()
    await lease.acquire("first")
    order: list[str] = []

    async def claim(name: str):
        await lease.acquire(name, timeout=5)
        order.append(name)
        await lease.release(name)

    tasks = [asyncio.create_task(claim(n)) for n in ("second", "third")]
    await asyncio.sleep(0.05)
    assert lease.info().waiters == 2
    await lease.release("first")
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert sorted(order) == ["second", "third"]
