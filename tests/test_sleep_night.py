"""A whole night, for real, on a machine with no GPU.

Every other test of the steps calls their bodies directly. This one runs the actual
pipeline: the runner records each step in the journal, spawns a child process, the child
rebuilds the project from the two paths it was handed, does its work, and reports back.

It is the test that catches the things direct calls cannot — a step whose module does not
import under `spawn`, a missing kwarg in the child contract, a step that writes outside
the project, a result that will not serialise through the journal.

There is no model here (`inference.responder: echo`), so the steps that need one say so
and the rest do their deterministic work. That is the same path a laptop with no CUDA
takes, and it has to end with a completed run and a report.
"""

from __future__ import annotations

import pytest

from gustav.config.load import load_config
from gustav.core.events import Events
from gustav.core.journal import Journal, RunStatus, StepStatus
from gustav.core.supervisor import Supervisor
from gustav.paths import Paths
from gustav.sleep.pipeline import PipelineRunner, default_pipeline
from gustav.store.db import open_db

CONFIG = """
instance:
  data_dir: var
logging:
  console: false
  level: DEBUG
inference:
  responder: echo          # no GPU, no weights: the steps that need a model say so
sleep:
  simulated_steps: [expansion, speed]
  simulated_step_seconds: 0.02
  heldout_fraction: 0.0
supervisor:
  heartbeat_timeout_seconds: 30
  step_timeout_seconds: 120
  max_attempts: 1
  retry_backoff_seconds: 0.05
  memory_limit_mb: 0
fetcher:
  enabled: false
  local_sources: ["notes"]
backup:
  location: backups
  min_free_mb: 1
"""


@pytest.fixture
def night(tmp_path, repo_root):
    root = tmp_path / "project"
    (root / "profiles").mkdir(parents=True)
    (root / "modules").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "config.yaml").write_text(CONFIG, encoding="utf-8")
    import shutil

    shutil.copy(repo_root / "modules" / "registry.yaml", root / "modules" / "registry.yaml")
    shutil.copy(repo_root / "docs" / "sources.yaml", root / "docs" / "sources.yaml")

    notes = root / "notes"
    notes.mkdir()
    (notes / "lifetime.md").write_text(
        "# Lifetimes\n\n## Elision rules\n\n"
        + ("Every reference in Rust has a lifetime, and most of them are inferred. " * 8)
        + "\n\n```\nfn longest<'a>(x: &'a str) -> &'a str { x }\n```\n",
        encoding="utf-8",
    )

    config, paths = load_config(root=root)
    paths.ensure()
    db = open_db(paths.db)

    # A day's worth of experience: a question, a correction, a stated fact.
    session = db.insert("sessions", {"source": "test"})
    for question, answer in [
        ("why does the borrow checker reject this lifetime?", "no idea"),
        ("my editor is helix", "noted"),
    ]:
        db.insert("turns", {"session_id": session, "role": "user", "content": question})
        assistant = db.insert(
            "turns", {"session_id": session, "role": "assistant", "content": answer}
        )
    db.insert(
        "feedback",
        {
            "turn_id": assistant,
            "kind": "correct",
            "payload": "the reference outlives the value it borrows",
            "source": "user",
        },
    )
    yield config, paths, db
    db.close()


@pytest.mark.asyncio
async def test_the_whole_night_runs_in_real_child_processes(night):
    config, paths, db = night
    journal = Journal(db)
    events = Events(db)
    pipeline = default_pipeline(config)
    run = journal.create_run(trigger="test", steps=pipeline.specs)
    db.execute(
        "UPDATE sleep_runs SET run_dir = ? WHERE id = ?", (str(paths.run_dir(run.id)), run.id)
    )
    runner = PipelineRunner(
        config=config,
        journal=journal,
        supervisor=Supervisor(poll_interval=0.05),
        events=events,
        paths=paths,
    )
    outcome = await runner.run(journal.load_run(run.id), pipeline)

    assert outcome.status is RunStatus.COMPLETED, outcome.reason
    assert not outcome.failed, f"steps failed: {outcome.failed}"

    record = journal.load_run(run.id)
    assert all(step.status is StepStatus.DONE for step in record.steps)
    results = {step.name: step.result or {} for step in record.steps}

    # The backup rule: the night's first act is a verified snapshot.
    assert results["snapshot"]["verified"]

    # Consolidation turned the day into material, with the correction weighted.
    assert results["consolidation"]["corrections"] == 1
    assert results["consolidation"]["facts_new"] >= 1
    correction = db.query_one("SELECT * FROM training_examples WHERE kind = 'correction'")
    assert correction["weight"] == 10.0

    # Reflect found the topic from the correction, and study read the local document.
    assert results["reflect"]["topics"], "the correction should name a weak topic"
    assert results["fetch_docs"]["fetched"] >= 1
    assert results["study"]["examples"] >= 1
    cited = db.query_one("SELECT citation FROM training_examples WHERE kind = 'study'")
    assert cited["citation"].startswith("file://")

    # The steps that need a model said so rather than inventing something.
    assert "model" in results["practice"]["reason"]
    assert results["training"]["trained"] == 0
    assert "nothing was trained" in results["gate"]["reason"]

    # And the night wrote its report.
    report = paths.reports / f"run-{run.id:06d}.html"
    assert report.exists()
    assert "Sleep run" in report.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_dry_run_leaves_the_store_untouched(night):
    config, paths, db = night
    journal = Journal(db)
    pipeline = default_pipeline(config)
    run = journal.create_run(trigger="test", steps=pipeline.specs, dry_run=True)
    db.execute(
        "UPDATE sleep_runs SET run_dir = ? WHERE id = ?", (str(paths.run_dir(run.id)), run.id)
    )
    runner = PipelineRunner(
        config=config,
        journal=journal,
        supervisor=Supervisor(poll_interval=0.05),
        events=Events(db),
        paths=paths,
    )
    outcome = await runner.run(journal.load_run(run.id), pipeline)

    assert outcome.status is RunStatus.COMPLETED
    assert db.query_one("SELECT * FROM training_examples") is None
    assert db.query_one("SELECT * FROM facts") is None
    assert db.query_one("SELECT * FROM documents") is None
    assert not list(paths.reports.glob("run-*.html"))
    assert not (paths.root / "backups").exists(), "a dry run takes no snapshot either"
