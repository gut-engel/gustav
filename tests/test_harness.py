"""The harness, driven by a fake backend: routing, context, adapters, the eval hook."""

from __future__ import annotations

import pytest
import pytest_asyncio

from gustav.config.schema import Config
from gustav.core.events import Events, Telemetry
from gustav.hw.detect import simulated
from gustav.hw.ladder import load_ladder
from gustav.hw.service import HardwareService
from gustav.inference.backend import GenerationParams
from gustav.inference.harness import Harness
from gustav.inference.tiered_cache import Tier
from gustav.testing.fakes import FakeBackend, write_fake_adapter

pytestmark = pytest.mark.asyncio


@pytest.fixture
def hardware(paths, db, events):
    config = Config.model_validate({"logging": {"console": False}})
    service = HardwareService(
        config, paths, db, events, detector=simulated("small-12gb"), ladder=load_ladder()
    )
    service.ensure_plan()
    return service


@pytest.fixture
def backend():
    return FakeBackend(base_model="qwen3-4b")


@pytest_asyncio.fixture
async def harness(paths, db, events, hardware, backend, repo_root):
    import shutil

    paths.registry.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(repo_root / "modules" / "registry.yaml", paths.registry)
    instance = Harness(
        config=Config.model_validate({"logging": {"console": False}}),
        paths=paths,
        hardware=hardware,
        events=events,
        telemetry=Telemetry(db),
        db=db,
        backend=backend,
    )
    return instance


# --- loading --------------------------------------------------------------
async def test_loading_brings_up_the_backend(harness, backend):
    await harness.load()
    assert harness.loaded and backend.load_count == 1


async def test_unloading_releases_the_backend(harness, backend):
    await harness.load()
    await harness.unload()
    assert not harness.loaded and backend.unload_count == 1


async def test_active_adapters_are_loaded_at_startup(harness, paths, backend):
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    harness.adapters.activate("coding", "v001")
    await harness.load()
    assert backend.loaded_adapters == {"coding": "v001"}


async def test_an_adapter_for_a_different_base_is_skipped(harness, paths, backend, events):
    """Adapters belong to the base they were trained on."""
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="some-other-model")
    harness.adapters.activate("coding", "v001")
    await harness.load()
    assert backend.loaded_adapters == {}
    assert any(e["kind"] == "inference.adapter_mismatch" for e in events.recent())


async def test_a_broken_adapter_does_not_stop_the_model(harness, paths):
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    harness.adapters.activate("coding", "v001")
    harness.backend.fail_adapter = "coding"
    await harness.load()
    assert harness.loaded
    assert any("did not load" in note for note in harness.describe()["notes"])


# --- answering ------------------------------------------------------------
async def test_a_reply_carries_the_numbers_and_the_modules(harness):
    await harness.load()
    reply = await harness.respond(session_id=1, message="rust borrow checker", history=[])
    assert reply.base_model == "qwen3-4b"
    assert reply.tokens_per_second == 42.0
    assert "rust" in reply.meta["modules"]


async def test_the_router_picks_the_adapters_that_are_applied(harness, paths, backend):
    for module in ("coding", "rust"):
        write_fake_adapter(paths.adapters, module, "v001", base_model="qwen3-4b")
        harness.adapters.activate(module, "v001")
    await harness.load()
    await harness.respond(session_id=1, message="why does cargo reject this lifetime?", history=[])
    assert "rust" in backend.active_adapters


async def test_explicit_adapters_override_the_router(harness, paths, backend):
    for module in ("coding", "rust"):
        write_fake_adapter(paths.adapters, module, "v001", base_model="qwen3-4b")
        harness.adapters.activate(module, "v001")
    await harness.load()
    await harness.respond(
        session_id=1, message="anything at all", history=[],
        params=GenerationParams(adapters=("coding",)),
    )
    assert set(backend.active_adapters) == {"coding"}


async def test_history_is_passed_through_and_trimmed(harness, backend):
    await harness.load()
    history = [{"role": "user", "content": f"old {i} " + "word " * 200} for i in range(60)]
    await harness.respond(session_id=1, message="the new question", history=history)
    prompt = backend.prompts[-1]
    assert prompt[-1]["content"] == "the new question"
    assert len(prompt) < len(history)


async def test_streaming_yields_pieces_and_still_records(harness):
    await harness.load()
    chunks = [
        chunk.text
        async for chunk in harness.stream(session_id=1, message="hello", history=[])
    ]
    assert "".join(chunks).startswith("fake answer")
    assert harness.last_generation() is not None


async def test_telemetry_is_recorded(harness, db):
    await harness.load()
    await harness.respond(session_id=1, message="hello", history=[])
    metrics = {r["metric"] for r in db.query("SELECT metric FROM telemetry")}
    assert "inference.tokens_per_second" in metrics
    assert "router.decision" in metrics


async def test_an_uncovered_topic_is_logged(harness, events):
    await harness.load()
    await harness.respond(session_id=1, message="what is the capital of peru", history=[])
    assert any(e["kind"] == "router.uncovered" for e in events.recent(limit=100))


# --- hot swap and rollback ------------------------------------------------
async def test_activating_a_new_version_swaps_it_in_without_a_restart(harness, paths, backend):
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    harness.adapters.activate("coding", "v001")
    await harness.load()
    write_fake_adapter(paths.adapters, "coding", "v002", base_model="qwen3-4b")

    await harness.activate_adapter("coding", "v002")
    assert backend.loaded_adapters["coding"] == "v002"
    assert harness.adapters.active("coding").version == "v002"
    assert backend.load_count == 1, "the base model must not be reloaded"


async def test_rollback_returns_to_the_previous_version(harness, paths, backend, events):
    for version in ("v001", "v002"):
        write_fake_adapter(paths.adapters, "coding", version, base_model="qwen3-4b")
        harness.adapters.activate("coding", version)
    await harness.load()

    rolled = await harness.rollback_adapter("coding")
    assert rolled.version == "v001"
    assert backend.loaded_adapters["coding"] == "v001"
    assert any(e["kind"] == "adapter.rolled_back" for e in events.recent())


async def test_adapter_status_reports_versions_and_tiers(harness, paths):
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    harness.adapters.activate("coding", "v001")
    await harness.load()
    status = harness.adapter_status()
    assert status["modules"]["coding"]["active"] == "v001"
    assert status["modules"]["coding"]["loaded"] is True
    assert status["modules"]["coding"]["tier"] == str(Tier.VRAM)
    assert status["cache"]["vram"]["budget_mb"] > 0


# --- the evaluation hook --------------------------------------------------
async def test_evaluate_runs_the_prompts(harness):
    await harness.load()
    results = await harness.evaluate(
        [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]
    )
    assert len(results) == 2
    assert "one" in results[0].text


async def test_evaluate_uses_the_candidate_then_restores_the_active_one(harness, paths, backend):
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    write_fake_adapter(paths.adapters, "coding", "v002", base_model="qwen3-4b")
    harness.adapters.activate("coding", "v001")
    await harness.load()

    results = await harness.evaluate(
        [[{"role": "user", "content": "test"}]], candidate=("coding", "v002")
    )
    assert results[0].adapters == {"coding": "v002"}
    # The live adapter is back afterwards.
    assert backend.loaded_adapters["coding"] == "v001"
    assert harness.adapters.active("coding").version == "v001"


async def test_the_active_adapter_is_restored_even_if_the_candidate_fails(harness, paths, backend):
    """The regression gate depends on this: a bad candidate must not poison the model."""
    write_fake_adapter(paths.adapters, "coding", "v001", base_model="qwen3-4b")
    write_fake_adapter(paths.adapters, "coding", "v002", base_model="qwen3-4b")
    harness.adapters.activate("coding", "v001")
    await harness.load()

    original_generate = backend.generate

    async def explode(*args, **kwargs):
        raise RuntimeError("the candidate produced garbage")

    backend.generate = explode
    with pytest.raises(RuntimeError, match="garbage"):
        await harness.evaluate(
            [[{"role": "user", "content": "test"}]], candidate=("coding", "v002")
        )
    backend.generate = original_generate
    assert backend.loaded_adapters["coding"] == "v001"
    assert harness.adapters.active("coding").version == "v001"


async def test_evaluate_is_deterministic_by_default(harness):
    await harness.load()
    await harness.evaluate([[{"role": "user", "content": "x"}]])
    assert harness.backend.params[-1].temperature == 0.0


# --- reporting ------------------------------------------------------------
async def test_describe_before_loading_says_so(harness):
    described = harness.describe()
    assert described["loaded"] is False
    assert "not been loaded" in described["notes"][0]


async def test_describe_after_loading_has_the_details(harness):
    await harness.load()
    described = harness.describe()
    assert described["base_model"] == "qwen3-4b"
    assert described["modules"]
    assert "adapter_cache" in described


async def test_health_is_honest_before_and_after(harness):
    """"Not loaded" is not "unhealthy": the daemon health-checks before it loads."""
    assert harness.health()[0] is True
    await harness.load()
    assert harness.health()[0] is True
    await harness.unload()
    ok, message = harness.health()
    assert ok is True and "nothing has failed" in message


async def test_a_failed_load_is_unhealthy(harness):
    harness.backend.fail_to_load = "no weights on this machine"
    with pytest.raises(Exception, match="no weights"):
        await harness.load()
    ok, message = harness.health()
    assert not ok and "no weights" in message
    assert any("failed" in note for note in harness.describe()["notes"])
