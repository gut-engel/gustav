"""The gate: measure the candidate, and promote only if it is better.

This is the step the whole design exists to make possible. A system that trains on its
own output every night will drift, and the only defence is a measurement it cannot talk
its way past:

* **Both suites, both versions, same prompts, greedy.** The candidate and the currently
  active weights answer identical questions at temperature 0, in one process, in the
  same session. Comparing tonight's number against a number from last week would
  measure the weather.
* **Better on held-out, and not worse on what it already knew.** Improvement on the
  held-out split has to clear ``gate.min_improvement``; regression on the retention
  suite has to stay within ``gate.max_regression``. Both, not either.
* **Too few items means skip, not promote.** Below ``gate.min_items`` there is no
  measurement, and "we did not measure it" must never be a reason to switch weights on.
* **Promotion is a guarded symlink swap.** One `os.replace`, inside
  ``guard.change(...)``, with a verified snapshot already taken. A crash at any instant
  leaves the previous version live.
* **At most ``gate.max_promotions_per_run`` modules change per night.** If something
  breaks tomorrow, the cause should be a single, named change.

A rejected candidate is kept on disk, unpromoted, with its gate result recorded — so
"why did it not promote" is answerable, and `ai adapters` shows it as a version that
exists and was not chosen.
"""

from __future__ import annotations

import json
from typing import Any

from gustav.inference.adapters import AdapterStore
from gustav.learn.bench import BenchItem, BenchmarkStore, score_answer, seed_regression_suite
from gustav.sandbox import Sandbox
from gustav.sleep.stages._backup import weight_guard
from gustav.sleep.stages._base import StepEnv, sleep_step


def _sandbox(env: StepEnv) -> Sandbox:
    cfg = env.config.sandbox
    return Sandbox(
        python=cfg.python,
        timeout_seconds=cfg.timeout_seconds,
        memory_limit_mb=cfg.memory_limit_mb,
        max_output_bytes=cfg.max_output_bytes,
        workdir=env.paths.tmp,
        isolate_env=cfg.isolate_env,
    )


def heldout_items(env: StepEnv, module: str, *, limit: int) -> list[BenchItem]:
    """The held-out split, as benchmark items. Never trained on, by construction."""
    rows = env.examples.heldout(module=module, limit=limit)
    items = [
        BenchItem(
            prompt=row["prompt"],
            kind="qa",
            expected=row["completion"],
            verifier={"match": "prose"},
            module=module,
            suite="heldout",
        )
        for row in rows
    ]
    items += BenchmarkStore(env.db).suite("heldout", module=module, limit=limit)
    return items[:limit]


def measure(
    env: StepEnv,
    model: Any,
    items: list[BenchItem],
    *,
    candidate: tuple[str, str] | None,
    sandbox: Sandbox,
    max_tokens: int,
) -> tuple[float, list[dict[str, Any]]]:
    if not items:
        return 0.0, []
    answers = model.evaluate(
        [item.prompt for item in items],
        candidate=candidate,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    detail: list[dict[str, Any]] = []
    passed = 0
    for item, answer in zip(items, answers, strict=True):
        ok, why = score_answer(item, answer, sandbox=sandbox)
        passed += int(ok)
        detail.append({"prompt": item.prompt[:120], "passed": ok, "why": why[:160]})
    return passed / len(items), detail


def _record(
    env: StepEnv,
    *,
    module: str,
    candidate: str,
    baseline: str,
    suite: str,
    items: int,
    candidate_score: float | None,
    baseline_score: float | None,
    verdict: str,
    detail: Any,
) -> None:
    env.db.insert(
        "gate_results",
        {
            "run_id": env.run_id,
            "module": module,
            "candidate": candidate,
            "baseline": baseline,
            "suite": suite,
            "items": items,
            "candidate_score": candidate_score,
            "baseline_score": baseline_score,
            "delta": (
                None
                if candidate_score is None or baseline_score is None
                else round(candidate_score - baseline_score, 4)
            ),
            "verdict": verdict,
            "detail": json.dumps(detail, default=str)[:20000],
        },
    )


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    cfg = env.config.gate
    adapters = AdapterStore(env.paths.adapters)
    result: dict[str, Any] = {"candidates": {}, "promoted": [], "rejected": [], "skipped": []}

    trained = env.db.query(
        "SELECT module, version FROM training_runs WHERE run_id = ?"
        " AND status IN ('trained', 'preempted') AND version IS NOT NULL",
        (env.run_id,),
    )
    if not trained:
        result["reason"] = "nothing was trained tonight"
        return result
    if not cfg.enabled:
        # Deliberately loud: a disabled gate means new weights go live unmeasured.
        env.log(
            "gate.enabled is false: candidates will not be promoted, because promoting "
            "without a measurement is the one thing this step exists to prevent",
            level="warning",
        )
        result["reason"] = "gate.enabled is false; candidates were left unpromoted"
        return result

    model = env.model()
    if not model.available:
        result["reason"] = model.error or "no model available to measure with"
        env.log(f"cannot run the gate: {result['reason']}", level="warning")
        return result

    sandbox = _sandbox(env)
    guard = weight_guard(env)
    promotions = 0

    for _index, row in env.each(list(trained), label="candidate", checkpoint_key="candidate_index"):
        module, version = row["module"], row["version"]
        active = adapters.active(module)
        baseline_version = active.version if active else "(base model)"

        seeded = seed_regression_suite(
            env.db,
            module=module,
            limit=cfg.max_items,
            run_id=env.run_id,
            exclude_run_id=env.run_id,
        )
        held = heldout_items(env, module, limit=cfg.max_items)
        regression = BenchmarkStore(env.db).suite("regression", module=module, limit=cfg.max_items)
        env.progress(
            f"{module} {version}: {len(held)} held-out and {len(regression)} retention items",
            heldout=len(held),
            regression=len(regression),
        )

        if len(held) < cfg.min_items:
            reason = (
                f"only {len(held)} held-out items; gate.min_items is {cfg.min_items}, "
                f"so there is nothing to measure"
            )
            _record(
                env, module=module, candidate=version, baseline=baseline_version,
                suite="heldout", items=len(held), candidate_score=None,
                baseline_score=None, verdict="skip", detail={"reason": reason},
            )
            env.db.execute(
                "UPDATE training_runs SET status = 'skipped', detail = ? WHERE run_id = ?"
                " AND module = ?",
                (json.dumps({"gate": reason}), env.run_id, module),
            )
            result["skipped"].append(module)
            result["candidates"][module] = {"verdict": "skip", "reason": reason,
                                            "version": version}
            env.log(f"{module} {version}: {reason}", level="warning")
            continue

        if env.dry_run:
            result["candidates"][module] = {
                "verdict": "dry-run", "heldout_items": len(held),
                "regression_items": len(regression), "seeded": seeded,
            }
            continue

        base_held, base_held_detail = measure(
            env, model, held, candidate=None, sandbox=sandbox, max_tokens=cfg.max_new_tokens
        )
        cand_held, cand_held_detail = measure(
            env, model, held, candidate=(module, version), sandbox=sandbox,
            max_tokens=cfg.max_new_tokens,
        )
        base_reg, _ = measure(
            env, model, regression, candidate=None, sandbox=sandbox,
            max_tokens=cfg.max_new_tokens,
        )
        cand_reg, cand_reg_detail = measure(
            env, model, regression, candidate=(module, version), sandbox=sandbox,
            max_tokens=cfg.max_new_tokens,
        )

        improvement = cand_held - base_held
        forgotten = base_reg - cand_reg
        passed = improvement >= cfg.min_improvement and forgotten <= cfg.max_regression
        verdict = "pass" if passed else "fail"
        reason = (
            f"held-out {base_held:.2f} -> {cand_held:.2f} "
            f"({improvement:+.2f}, needs {cfg.min_improvement:+.2f}); "
            f"retention {base_reg:.2f} -> {cand_reg:.2f} "
            f"({-forgotten:+.2f}, allowed {-cfg.max_regression:+.2f})"
        )

        _record(
            env, module=module, candidate=version, baseline=baseline_version, suite="heldout",
            items=len(held), candidate_score=round(cand_held, 4),
            baseline_score=round(base_held, 4), verdict=verdict,
            detail={"candidate": cand_held_detail[:40], "baseline": base_held_detail[:40]},
        )
        if regression:
            _record(
                env, module=module, candidate=version, baseline=baseline_version,
                suite="regression", items=len(regression), candidate_score=round(cand_reg, 4),
                baseline_score=round(base_reg, 4),
                verdict="pass" if forgotten <= cfg.max_regression else "fail",
                detail={"candidate": cand_reg_detail[:40]},
            )

        entry = {
            "version": version,
            "baseline": baseline_version,
            "verdict": verdict,
            "heldout": {"baseline": round(base_held, 4), "candidate": round(cand_held, 4),
                        "items": len(held)},
            "regression": {"baseline": round(base_reg, 4), "candidate": round(cand_reg, 4),
                           "items": len(regression)},
            "reason": reason,
        }
        result["candidates"][module] = entry

        if not passed:
            result["rejected"].append(module)
            env.db.execute(
                "UPDATE training_runs SET status = 'rejected', detail = ? WHERE run_id = ?"
                " AND module = ?",
                (json.dumps({"gate": reason}), env.run_id, module),
            )
            env.events.emit(
                "sleep.gate.rejected",
                f"{module} {version} was not promoted: {reason}",
                level="warning",
                run_id=env.run_id, module=module, version=version,
            )
            continue

        if promotions >= cfg.max_promotions_per_run:
            entry["verdict"] = "held"
            entry["reason"] = (
                f"passed, but gate.max_promotions_per_run is "
                f"{cfg.max_promotions_per_run} and that is already used tonight"
            )
            result["skipped"].append(module)
            env.events.emit(
                "sleep.gate.held",
                f"{module} {version} passed but was held back: {entry['reason']}",
                run_id=env.run_id, module=module, version=version,
            )
            continue

        # The only place in the night where what is served changes.
        with guard.change(
            f"promote {module} {version}",
            sources=("adapters",),
            run_id=env.run_id,
            step=env.step,
            meta={"module": module, "version": version, "gate": reason},
        ) as change:
            link = adapters.module_dir(module) / "active"
            change.activate(link, version)
            meta_path = adapters.module_dir(module) / version / "meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            meta.update(
                {
                    "promoted": True,
                    "promoted_run_id": env.run_id,
                    "benchmark_score": round(cand_held, 4),
                    "baseline_score": round(base_held, 4),
                    "gate": reason,
                }
            )
            adapters.write_meta(module, version, meta)

        promotions += 1
        result["promoted"].append(module)
        env.db.execute(
            "UPDATE training_runs SET status = 'promoted', detail = ? WHERE run_id = ?"
            " AND module = ?",
            (json.dumps({"gate": reason}), env.run_id, module),
        )
        env.events.emit(
            "sleep.gate.promoted",
            f"{module} {version} promoted: {reason}",
            run_id=env.run_id, module=module, version=version,
            snapshot=change.snapshot_id,
        )

    env.write_json("gate.json", result)
    return result
