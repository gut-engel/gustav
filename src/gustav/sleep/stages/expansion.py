"""Expansion: grow only if the measurements demand it.

Section 8. The step reads what the store already knows — requests the router could not
place, modules whose gate results have stopped moving, adapters nobody uses — turns
those into proposals that the hardware budget has to allow, and then does the cheap
reversible ones and *asks* about the rest.

Three rules make this safe enough to leave running:

* **The budget is derived, not configured.** ``ExpansionBudget`` comes from the card:
  how many adapters fit beside inference, the largest LoRA rank that fits, how much
  warm cache there is. A proposal that does not fit is recorded as refused with the
  arithmetic, which is more useful than silence.
* **One structural change per night**, so tomorrow's "what broke" has one answer.
* **The expensive ones wait for you.** Archiving a module changes what the system can
  do; merging an adapter into the base rewrites the weights everything else is built
  on. Both go to `ai approvals` with their evidence instead of happening.
"""

from __future__ import annotations

from typing import Any

from gustav.expansion.actions import Expander
from gustav.expansion.proposals import propose
from gustav.expansion.triggers import measure
from gustav.hw.service import HardwareService
from gustav.inference.adapters import AdapterStore
from gustav.inference.registry import load_registry
from gustav.sleep.stages._backup import weight_guard
from gustav.sleep.stages._base import StepEnv, sleep_step


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    cfg = env.config.expansion
    result: dict[str, Any] = {
        "triggers": [],
        "proposals": [],
        "applied": [],
        "pending": [],
        "refused": [],
    }
    if not cfg.enabled:
        result["reason"] = "expansion.enabled is false"
        return result

    registry = load_registry(env.paths.registry)
    adapters = AdapterStore(env.paths.adapters)
    hardware = HardwareService(env.config, env.paths, env.db, env.events)
    plan = hardware.stored_plan()
    if plan is None:
        result["reason"] = "there is no hardware plan, so there is no expansion budget"
        return result

    triggers = measure(env.db, registry, cfg, adapters=adapters)
    result["triggers"] = [
        {"kind": t.kind, "subject": t.subject, "strength": round(t.strength, 3),
         "rationale": t.rationale}
        for t in triggers
    ]
    if not triggers:
        result["reason"] = "nothing measured asks for more capacity"
        return result

    proposals = propose(
        triggers,
        registry=registry,
        budget=plan.expansion,
        config=cfg,
        plan_rank=plan.lora_rank,
        adapter_count=len(adapters.modules()),
    )
    result["proposals"] = [p.describe() for p in proposals]
    result["budget"] = plan.expansion.describe()

    expander = Expander(
        paths=env.paths,
        config=env.config,
        db=env.db,
        events=env.events,
        guard=weight_guard(env),
        adapters=adapters,
    )

    if env.dry_run:
        result["dry_run"] = True
        return result

    # Resuming by *name*, not by position: the triggers are re-measured when the step
    # restarts, and the proposal that was third last time may not be third now — an
    # index would then apply something nobody looked at.
    handled: set[str] = set(env.checkpoint.get("handled") or [])
    changes = 0
    for _index, proposal in env.each(proposals, label="proposal"):
        key = f"{proposal.kind}:{proposal.subject}"
        if key in handled:
            continue
        handled.add(key)
        env.save({"handled": sorted(handled)})
        if not proposal.applicable:
            expander.record(proposal, run_id=env.run_id, status="refused",
                            detail=proposal.refused or "")
            result["refused"].append(proposal.describe())
            env.log(proposal.describe())
            continue
        if proposal.needs_approval:
            approval_id = expander.file_approval(proposal, run_id=env.run_id)
            result["pending"].append(
                {"approval": approval_id, "kind": proposal.kind, "subject": proposal.subject}
            )
            continue
        if changes >= cfg.max_changes_per_run:
            expander.record(
                proposal, run_id=env.run_id, status="proposed",
                detail=(
                    f"held back: expansion.max_changes_per_run is "
                    f"{cfg.max_changes_per_run} and that is used tonight"
                ),
            )
            result["refused"].append(
                f"{proposal.kind} {proposal.subject}: held back until tomorrow"
            )
            continue

        applied = expander.apply(
            proposal, run_id=env.run_id, progress=lambda message: env.progress(message)
        )
        if applied.ok:
            changes += 1
            result["applied"].append(
                {"kind": proposal.kind, "subject": proposal.subject,
                 "detail": applied.detail, "snapshot": applied.snapshot_id}
            )
        else:
            result["refused"].append(applied.describe())

    env.write_json("expansion.json", result)
    if result["applied"] or result["pending"]:
        env.events.emit(
            "sleep.expansion",
            f"{len(result['applied'])} change(s) applied, "
            f"{len(result['pending'])} waiting for approval",
            run_id=env.run_id,
            applied=len(result["applied"]),
            pending=len(result["pending"]),
        )
    return result
