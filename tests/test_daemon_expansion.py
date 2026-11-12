"""Expansion through the running daemon: approve, apply, revert.

The daemon is the only thing that may apply an approved proposal while it is serving,
and the properties that matter are the boring ones: the approval queue dispatches on
the approval's kind, the live router is told about a registry change rather than finding
out at the next restart, and a revert puts the registry back.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from gustav.config.load import load_config
from gustav.daemon import Daemon
from gustav.inference.registry import load_registry

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def daemon(project: Path, repo_root: Path) -> AsyncIterator[Daemon]:
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    config, paths = load_config(project / "config.yaml")
    instance = Daemon(config, paths)
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown("test finished")


@pytest_asyncio.fixture
async def client(daemon: Daemon) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.AsyncHTTPTransport(uds=str(daemon.socket_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://gustav") as http:
        yield http


def pending_proposal(daemon: Daemon, *, kind: str = "new_module", subject: str = "kubernetes"):
    """File an approval the way the night's expansion step would."""
    from gustav.expansion.proposals import Proposal

    proposal = Proposal(
        kind=kind,
        subject=subject,
        rationale="five requests nothing covered were about kubernetes",
        evidence={"requests": 5},
        payload={
            "name": subject,
            "module": subject,
            "description": "Kubernetes manifests, pods and services",
            "keywords": ["kubernetes", "pod", "manifest"],
            "lora_rank": 16,
        },
        needs_approval=True,
    )
    return daemon._expander().file_approval(proposal, run_id=None)


async def test_the_approval_queue_shows_an_expansion_with_its_reason(daemon, client):
    approval_id = pending_proposal(daemon)
    payload = (await client.get("/approvals")).json()
    entry = next(a for a in payload["approvals"] if a["id"] == approval_id)
    assert entry["kind"] == "expansion"
    assert entry["subject"] == "new_module: kubernetes"
    assert "kubernetes" in entry["rationale"]


async def test_approving_an_expansion_applies_it_and_tells_the_router(daemon, client):
    approval_id = pending_proposal(daemon)
    result = (
        await client.post(f"/approvals/{approval_id}", json={"approve": True})
    ).json()

    assert result["approved"] and result["kind"] == "new_module"
    assert result["snapshot"], "a verified snapshot was taken first"
    assert "kubernetes" in load_registry(daemon.paths.registry)
    row = daemon.db.query_one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
    assert row["status"] == "approved" and row["snapshot_id"] == result["snapshot"]


async def test_rejecting_an_expansion_changes_nothing(daemon, client):
    approval_id = pending_proposal(daemon)
    result = (
        await client.post(
            f"/approvals/{approval_id}", json={"approve": False, "note": "not now"}
        )
    ).json()
    assert result["rejected"]
    assert "kubernetes" not in load_registry(daemon.paths.registry)
    log = daemon.db.query_one(
        "SELECT * FROM expansion_log WHERE approval_id = ?", (approval_id,)
    )
    assert log["status"] == "rejected" and log["detail"] == "not now"


async def test_approving_twice_is_refused(daemon, client):
    approval_id = pending_proposal(daemon)
    await client.post(f"/approvals/{approval_id}", json={"approve": True})
    second = await client.post(f"/approvals/{approval_id}", json={"approve": True})
    assert second.status_code == 400
    assert "already approved" in second.json()["detail"]


async def test_an_unknown_approval_is_a_404(daemon, client):
    response = await client.post("/approvals/999", json={"approve": True})
    assert response.status_code == 404


async def test_the_expansion_view_reports_triggers_and_history(daemon, client):
    session = daemon.db.insert("sessions", {"source": "test"})
    # expansion.min_uncovered_requests defaults to 8: fewer than that is a bad day.
    for index in range(9):
        turn_id = daemon.db.insert(
            "turns",
            {"session_id": session, "role": "user",
             "content": f"a kubernetes manifest question {index}"},
        )
        daemon.db.insert(
            "telemetry",
            {"metric": "router.decision", "value": 0.1,
             "tags": json.dumps({"uncovered": True, "turn_id": turn_id})},
        )
    approval_id = pending_proposal(daemon)

    payload = (await client.get("/expansion")).json()
    assert payload["enabled"] is True
    assert "general" in payload["modules"]
    assert any(t["kind"] == "uncovered_topics" for t in payload["triggers"])
    assert any(item["approval_id"] == approval_id for item in payload["history"])


async def test_an_applied_expansion_can_be_reverted_through_the_daemon(daemon, client):
    approval_id = pending_proposal(daemon)
    await client.post(f"/approvals/{approval_id}", json={"approve": True})
    assert "kubernetes" in load_registry(daemon.paths.registry)

    result = (await client.post("/expansion/revert")).json()
    assert result["kind"] == "new_module"
    assert "kubernetes" not in load_registry(daemon.paths.registry)
    assert result["undo_snapshot"]


async def test_reverting_with_nothing_to_revert_is_a_404(daemon, client):
    response = await client.post("/expansion/revert")
    assert response.status_code == 404
