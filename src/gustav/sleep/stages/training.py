"""Training: a new adapter version per module, written but not switched on.

The step that actually changes weights, so it is the step with the most rules.

* **It goes through the guard.** Every module's training runs inside
  ``guard.change(...)``, which takes and *verifies* a snapshot before the body runs and
  restores from it if the body raises. No snapshot, no training — section 9.
* **It writes a new version, it does not promote one.** The adapter lands in
  ``adapters/<module>/vNNN`` with the ``active`` symlink untouched, so the model being
  served is unaffected no matter what comes out. Only `gate` may make it live.
* **It can be interrupted at any step.** A user message mid training saves the adapter,
  records how far it got, and gives the card back; the next night resumes from there
  rather than starting over.
* **It records what it trained on.** Which examples, in what proportions, with which
  replay — in ``training_runs`` and in the adapter's own ``meta.json``. An adapter whose
  dataset cannot be named afterwards is not reviewable.
"""

from __future__ import annotations

import shutil
from typing import Any

from gustav.hw.service import HardwareService
from gustav.inference.adapters import AdapterStore
from gustav.sleep.stages._backup import weight_guard
from gustav.sleep.stages._base import StepEnv, sleep_step
from gustav.train.dataset import build_mix
from gustav.train.lora import TrainOutcome, train_adapter, write_meta


def _record_start(env: StepEnv, module: str, version: str, mix: Any) -> int:
    return env.db.insert(
        "training_runs",
        {
            "run_id": env.run_id,
            "module": module,
            "version": version,
            "examples": mix.size,
            "replay": mix.replay,
        },
    )


def _record_finish(env: StepEnv, row_id: int, outcome: TrainOutcome, status: str) -> None:
    import json

    env.db.execute(
        "UPDATE training_runs SET finished_at = datetime('now'), steps = ?,"
        " loss_start = ?, loss_end = ?, seconds = ?, status = ?, detail = ?"
        " WHERE id = ?",
        (
            outcome.steps,
            outcome.loss_start,
            outcome.loss_end,
            outcome.seconds,
            status,
            json.dumps(outcome.detail, default=str),
            row_id,
        ),
    )


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    cfg = env.config.training
    result: dict[str, Any] = {"modules": [], "trained": 0, "skipped": 0, "candidates": {}}

    if not cfg.enabled:
        result["reason"] = "training.enabled is false"
        return result
    if not env.config.hardware.allow_training:
        result["reason"] = "hardware.allow_training is false (this machine only serves)"
        return result

    hardware = HardwareService(env.config, env.paths, env.db, env.events)
    plan = hardware.stored_plan()
    if plan is None:
        result["reason"] = "there is no hardware plan yet"
        return result
    if not plan.training_allowed:
        result["reason"] = "the hardware plan says this machine cannot train"
        return result

    ladder = hardware.effective_ladder()
    spec = ladder.model(plan.base_model)
    quant = ladder.quant(plan.quantization)
    adapters = AdapterStore(env.paths.adapters)
    guard = weight_guard(env)

    candidates = cfg.modules or sorted(env.examples.modules_with_material(cfg.min_examples))
    result["modules"] = list(candidates)
    if not candidates:
        result["reason"] = (
            f"no module has {cfg.min_examples} new examples "
            f"(training.min_examples)"
        )
        return result
    if env.dry_run:
        result["dry_run"] = True
        for module in candidates:
            mix = build_mix(
                env.examples,
                module=module,
                replay_fraction=env.config.sleep.replay_fraction,
                limit=env.config.sleep.max_examples_per_run,
                verified_only=cfg.verified_only,
            )
            result["candidates"][module] = {
                "would_train": mix.fresh >= cfg.min_examples,
                "examples": mix.size,
                "fresh": mix.fresh,
                "replay": mix.replay,
                "by_kind": mix.by_kind,
                "version": adapters.next_version(module),
            }
        return result

    resumed_versions: dict[str, Any] = dict(env.checkpoint.get("versions") or {})

    for _index, module in env.each(candidates, label="module", checkpoint_key="module_index"):
        mix = build_mix(
            env.examples,
            module=module,
            replay_fraction=env.config.sleep.replay_fraction,
            limit=env.config.sleep.max_examples_per_run,
            verified_only=cfg.verified_only,
        )
        if mix.fresh < cfg.min_examples:
            result["skipped"] += 1
            result["candidates"][module] = {
                "trained": False,
                "reason": f"only {mix.fresh} new examples",
            }
            continue

        state = resumed_versions.get(module) or {}
        version = state.get("version") or adapters.next_version(module)
        resume_step = int(state.get("step") or 0)
        env.log(f"training {module} {version}: {mix.describe()}")
        row_id = _record_start(env, module, version, mix)
        target = adapters.module_dir(module) / version

        try:
            with guard.change(
                f"train {module} {version}",
                sources=("adapters",),
                run_id=env.run_id,
                step=env.step,
                meta={"module": module, "version": version, "examples": mix.size},
            ) as change:
                if resume_step and target.exists():
                    # Carry the partially trained adapter into this attempt's staging
                    # directory, so a resumed night continues rather than restarts.
                    shutil.copytree(target, change.staging, dirs_exist_ok=True)

                def progress(message: str, data: dict[str, Any]) -> None:
                    env.progress(message, **data)
                    step = data.get("checkpoint_step")
                    if step:
                        env.save(
                            {
                                "versions": {
                                    **resumed_versions,
                                    module: {"version": version, "step": int(step)},
                                }
                            }
                        )

                outcome = train_adapter(
                    mix=mix,
                    plan=plan,
                    spec=spec,
                    quant=quant,
                    models_dir=env.paths.models,
                    output_dir=change.staging,
                    config=cfg,
                    on_progress=progress,
                    should_stop=env.stopping,
                    resume_step=resume_step,
                )
                if outcome.path is None:
                    raise RuntimeError(f"training produced no adapter: {outcome.stopped_by}")
                write_meta(
                    change.staging,
                    module=module,
                    version=version,
                    plan=plan,
                    outcome=outcome,
                    mix=mix,
                    extra={"run_id": env.run_id, "promoted": False},
                )
                change.promote(change.staging, target)
        except Exception as exc:  # noqa: BLE001 - one module failing is not the night
            _record_finish(env, row_id, TrainOutcome(module=module), "failed")
            result["candidates"][module] = {"trained": False, "reason": str(exc)[:300]}
            env.log(f"training {module} failed: {exc}", level="warning")
            env.events.emit(
                "sleep.training.failed",
                f"training {module} failed: {exc}",
                level="warning",
                run_id=env.run_id,
                module=module,
            )
            continue

        _record_finish(
            env, row_id, outcome, "preempted" if outcome.preempted else "trained"
        )
        # Only the fresh examples are consumed: the replayed ones stay available, which
        # is what makes them replayable again next time.
        env.examples.mark_used(mix.fresh_ids, run_id=env.run_id)
        result["trained"] += 1
        result["candidates"][module] = {
            "trained": True,
            "version": version,
            "examples": outcome.examples,
            "fresh": mix.fresh,
            "replay": mix.replay,
            "steps": outcome.steps,
            "loss_start": outcome.loss_start,
            "loss_end": outcome.loss_end,
            "seconds": outcome.seconds,
            "preempted": outcome.preempted,
            "by_kind": mix.by_kind,
        }
        env.events.emit(
            "sleep.trained",
            outcome.describe(),
            run_id=env.run_id,
            module=module,
            version=version,
            steps=outcome.steps,
        )

        if outcome.preempted:
            resumed_versions[module] = {"version": version, "step": outcome.steps}
            env.save({"versions": resumed_versions})
            env.write_json("training.json", result)
            # Hand the card back at a clean boundary; the night resumes here.
            env.raise_if_stopping()
        else:
            resumed_versions.pop(module, None)
            env.save({"versions": resumed_versions})

    env.write_json("training.json", result)
    return result
