"""A whole night with real weights on a real GPU.

This is the test that exercises stage 5 the way it will actually run: the daemon's
pipeline, in child processes, with a model loaded from disk, generating, being verified
by the sandbox, training a LoRA, and measuring it against the weights that were being
served before.

It is deliberately sized to be *small* — the 0.6B model, a short sequence length, two
training steps, a handful of benchmark items — because the question here is whether the
night holds together end to end, not how good the resulting adapter is. Quality is not
asserted anywhere: a 0.6B model's answers are not reliable enough to assert on, and a
test that depends on them would fail for the wrong reasons.

Run with: pytest -m gpu
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu]

torch = pytest.importorskip("torch", reason="needs the ML extras")
pytest.importorskip("peft", reason="needs the ML extras")

MODEL_ID = "qwen3-0.6b"


def weights_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / MODEL_ID


pytestmark.append(pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"))
pytestmark.append(
    pytest.mark.skipif(
        not (weights_dir() / "config.json").exists(),
        reason=f"weights missing: ai model pull {MODEL_ID}",
    )
)

CONFIG = f"""
instance:
  data_dir: var
logging:
  console: false
  level: INFO
hardware:
  prefer_model: {MODEL_ID}
  min_tokens_per_second: 1.0
  train_seq_len: 192
  max_seq_len: 1024
inference:
  responder: auto
  max_tokens: 128
sleep:
  simulated_steps: [expansion, speed]
  heldout_fraction: 0.5      # a tiny night still needs something to measure with
  max_new_tokens: 128
supervisor:
  heartbeat_timeout_seconds: 120
  step_timeout_seconds: 900
  max_attempts: 1
  memory_limit_mb: 0
fetcher:
  enabled: false
  local_sources: ["notes"]
sandbox:
  timeout_seconds: 10
  memory_limit_mb: 1024
training:
  min_examples: 2
  max_steps: 2
  grad_accum: 1
  checkpoint_every_steps: 1
gate:
  min_items: 2
  max_items: 6
  max_new_tokens: 64
backup:
  location: backups
  min_free_mb: 1
"""


@pytest.fixture
def night(tmp_path, repo_root):
    from gustav.config.load import load_config
    from gustav.config.schema import HardwareConfig
    from gustav.hw.detect import detect
    from gustav.hw.ladder import ModelSpec, load_ladder
    from gustav.hw.planner import Planner
    from gustav.hw.service import META_ACTIVE_BASE, META_PLAN
    from gustav.store.db import open_db

    root = tmp_path / "project"
    (root / "profiles").mkdir(parents=True)
    (root / "modules").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "config.yaml").write_text(CONFIG, encoding="utf-8")
    shutil.copy(repo_root / "modules" / "registry.yaml", root / "modules" / "registry.yaml")
    # The weights are large: point at the ones already on disk rather than copying.
    (root / "models").symlink_to(repo_root / "models")

    notes = root / "notes"
    notes.mkdir()
    (notes / "python-regex.md").write_text(
        "# Regular expressions in Python\n\n"
        "## Compiling a pattern\n\n"
        "The re module compiles a pattern with re.compile, which returns a pattern "
        "object. Calling match on it tries the pattern at the start of the string, "
        "while search looks anywhere in the string. findall returns every match as a "
        "list of strings, and sub replaces matches with a replacement string.\n\n"
        "```\nimport re\npattern = re.compile(r'[0-9]+')\nprint(pattern.findall('a1 b22'))\n```\n",
        encoding="utf-8",
    )

    config, paths = load_config(root=root)
    paths.ensure()
    db = open_db(paths.db)

    ladder = load_ladder()
    base = ladder.model(MODEL_ID)
    spec = ModelSpec.from_hf_config(weights_dir() / "config.json", spec_id=MODEL_ID)
    spec = ModelSpec(
        **{
            **spec.__dict__,
            "rung": base.rung,
            "quantizations": base.quantizations,
            "label": base.label,
            "lora_targets": base.lora_targets,
        }
    )
    plan = Planner(
        detect(),
        HardwareConfig(
            prefer_model=MODEL_ID, min_tokens_per_second=1.0, train_seq_len=192,
            max_seq_len=1024,
        ),
        ladder=ladder.with_overrides([spec]),
    ).plan()
    db.set_meta(META_PLAN, plan.to_dict())
    db.set_meta(META_ACTIVE_BASE, plan.base_model)

    # A day's experience: a correction, a stated fact, and a question about regexes.
    session = db.insert("sessions", {"source": "test"})
    exchanges = [
        ("how do I find every number in a string with a regex in python?", "I am not sure"),
        ("my editor is helix", "noted"),
    ]
    assistant = 0
    for question, answer in exchanges:
        db.insert("turns", {"session_id": session, "role": "user", "content": question})
        assistant = db.insert(
            "turns", {"session_id": session, "role": "assistant", "content": answer}
        )
    db.insert(
        "feedback",
        {
            "turn_id": 2,
            "kind": "correct",
            "payload": "use re.findall(r'[0-9]+', text)",
            "source": "user",
        },
    )
    db.insert(
        "feedback",
        {"turn_id": assistant, "kind": "good", "source": "user"},
    )
    yield config, paths, db, plan
    db.close()
    torch.cuda.empty_cache()


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_a_real_night_end_to_end(night):
    from gustav.core.events import Events
    from gustav.core.journal import Journal, RunStatus, StepStatus
    from gustav.core.supervisor import Supervisor
    from gustav.sleep.pipeline import PipelineRunner, default_pipeline

    config, paths, db, plan = night
    journal = Journal(db)
    pipeline = default_pipeline(config)
    run = journal.create_run(trigger="gpu-test", steps=pipeline.specs)
    db.execute(
        "UPDATE sleep_runs SET run_dir = ? WHERE id = ?", (str(paths.run_dir(run.id)), run.id)
    )
    runner = PipelineRunner(
        config=config,
        journal=journal,
        supervisor=Supervisor(poll_interval=0.2),
        events=Events(db),
        paths=paths,
    )
    outcome = await runner.run(journal.load_run(run.id), pipeline)

    record = journal.load_run(run.id)
    results = {step.name: step.result or {} for step in record.steps}
    failures = {s.name: s.error for s in record.steps if s.status is StepStatus.FAILED}
    assert outcome.status is RunStatus.COMPLETED, f"{outcome.reason}; failures: {failures}"
    assert not failures, failures

    # The model was really there.
    assert results["consolidation"]["model"] is True
    assert results["practice"]["model"] is True

    # The day became material, and the correction was weighted and verified.
    assert results["consolidation"]["corrections"] == 1
    correction = db.query_one("SELECT * FROM training_examples WHERE kind = 'correction'")
    assert correction["weight"] == 10.0 and correction["verified"] == 1

    # It read the local document and produced cited examples from it.
    assert results["fetch_docs"]["fetched"] >= 1
    assert results["study"]["chunks"] >= 1
    cited = db.query(
        "SELECT citation FROM training_examples WHERE kind = 'study' AND citation IS NOT NULL"
    )
    assert all(row["citation"].startswith("file://") for row in cited)

    # Training either produced a candidate version or said why it did not.
    training = results["training"]
    assert "reason" in training or training["trained"] >= 0
    rows = db.query("SELECT * FROM training_runs WHERE run_id = ?", (run.id,))
    for row in rows:
        assert row["status"] in (
            "trained", "promoted", "rejected", "skipped", "preempted", "failed"
        )
        if row["status"] in ("trained", "promoted", "rejected", "skipped"):
            adapter = paths.adapters / row["module"] / row["version"]
            assert (adapter / "adapter_model.safetensors").exists()

    # Nothing was promoted without a recorded measurement.
    promoted = [r["module"] for r in rows if r["status"] == "promoted"]
    for module in promoted:
        gate_rows = db.query(
            "SELECT * FROM gate_results WHERE run_id = ? AND module = ? AND suite = 'heldout'",
            (run.id, module),
        )
        assert gate_rows and gate_rows[0]["verdict"] == "pass"
        link = paths.adapters / module / "active"
        assert link.is_symlink()
        change = db.query_one(
            "SELECT * FROM weight_changes WHERE operation LIKE ? ORDER BY id DESC",
            (f"promote {module}%",),
        )
        assert change["status"] == "done" and change["snapshot_id"]

    # Every weight change that was started was finished or undone — never left open.
    assert db.query_one(
        "SELECT COUNT(*) AS n FROM weight_changes WHERE status = 'in_progress'"
    )["n"] == 0

    # A compact account of what the night did, for when this fails or when you want
    # to see what a real run looks like: pytest -m gpu -s
    print("\n--- the night ---")
    for step in record.steps:
        summary = {
            key: value
            for key, value in (step.result or {}).items()
            if key not in ("step", "by_topic", "candidates", "documents", "skills", "topics")
        }
        print(f"{step.name:14} {str(step.status):10} {summary}")

    # And the night wrote its report.
    report = paths.reports / f"run-{run.id:06d}.html"
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "Sleep run" in html and "regex" in html.lower()
