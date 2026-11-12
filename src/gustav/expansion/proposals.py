"""Turning a measurement into a proposal the budget allows.

The triggers say what is true; this says what could be done about it, what it would
cost, and whether it is the kind of thing that may happen without being asked. Nothing
here touches the disk.

The budget comes from the hardware plan (``ExpansionBudget``), which is derived from the
card rather than configured: how many adapters fit in the VRAM left over after
inference, the largest LoRA rank that fits, how much warm cache there is. A proposal
that does not fit is not a proposal — it is recorded as refused, with the arithmetic,
because "it wanted to grow and could not" is worth seeing in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gustav.config.schema import ExpansionConfig
from gustav.expansion.triggers import Trigger
from gustav.hw.planner import ExpansionBudget
from gustav.inference.registry import Registry

#: LoRA parameters per rank, per module, as a rough function of the base's shape. Used
#: only to compare a proposal against the budget, never to size an allocation.
PARAMS_PER_RANK = 2_000_000


@dataclass
class Proposal:
    kind: str
    subject: str
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    needs_approval: bool = True
    refused: str | None = None

    @property
    def applicable(self) -> bool:
        return self.refused is None

    def describe(self) -> str:
        if self.refused:
            return f"{self.kind} {self.subject}: refused — {self.refused}"
        gate = "needs your approval" if self.needs_approval else "may run tonight"
        return f"{self.kind} {self.subject}: {self.rationale} ({gate})"


def _rank_for(registry: Registry, module: str, plan_rank: int) -> int:
    spec = registry.get(module)
    return spec.lora_rank or plan_rank


def propose(
    triggers: list[Trigger],
    *,
    registry: Registry,
    budget: ExpansionBudget,
    config: ExpansionConfig,
    plan_rank: int,
    adapter_count: int,
) -> list[Proposal]:
    out: list[Proposal] = []
    for trigger in triggers:
        proposal = _for_trigger(
            trigger,
            registry=registry,
            budget=budget,
            config=config,
            plan_rank=plan_rank,
            adapter_count=adapter_count,
        )
        if proposal is not None:
            out.append(proposal)
    return out


def _for_trigger(
    trigger: Trigger,
    *,
    registry: Registry,
    budget: ExpansionBudget,
    config: ExpansionConfig,
    plan_rank: int,
    adapter_count: int,
) -> Proposal | None:
    kind = {
        "uncovered_topics": "new_module",
        "saturated_module": "raise_rank",
        "unused_module": "prune_module",
        "merge_candidate": "merge_to_base",
    }.get(trigger.kind)
    if kind is None:
        return None

    allowed = {
        "new_module": config.allow_new_modules,
        "raise_rank": config.allow_rank_increase,
        "prune_module": config.allow_pruning,
        "merge_to_base": config.allow_base_merge,
    }[kind]
    needs_approval = kind in config.require_approval

    if kind == "new_module":
        name = _module_name(trigger.subject, registry)
        proposal = Proposal(
            kind=kind,
            subject=name,
            rationale=trigger.rationale,
            evidence=trigger.evidence,
            needs_approval=needs_approval,
            cost={
                "adapters_after": adapter_count + 1,
                "max_adapters": budget.max_adapters,
                "lora_rank": plan_rank,
                "vram_mb": round(budget.max_adapter_vram_mb / max(budget.max_adapters, 1), 1),
            },
            payload={
                "name": name,
                "description": _description_for(trigger),
                "keywords": trigger.evidence.get("keywords", [])[:12],
                "lora_rank": plan_rank,
            },
        )
        if not allowed:
            proposal.refused = "expansion.allow_new_modules is off"
        elif name in registry:
            proposal.refused = f"a module called {name!r} already exists"
        elif adapter_count + 1 > budget.max_adapters:
            proposal.refused = (
                f"the hardware budget allows {budget.max_adapters} adapters and "
                f"{adapter_count} already exist"
            )
        return proposal

    if kind == "raise_rank":
        current = _rank_for(registry, trigger.subject, plan_rank)
        target = current + config.rank_step
        proposal = Proposal(
            kind=kind,
            subject=trigger.subject,
            rationale=trigger.rationale,
            evidence=trigger.evidence,
            needs_approval=needs_approval,
            cost={
                "rank": current,
                "target_rank": target,
                "max_lora_rank": budget.max_lora_rank,
                "params": target * PARAMS_PER_RANK,
                "max_total_adapter_params": budget.max_total_adapter_params,
            },
            payload={"module": trigger.subject, "lora_rank": target, "previous": current},
        )
        if not allowed:
            proposal.refused = "expansion.allow_rank_increase is off"
        elif target > budget.max_lora_rank:
            proposal.refused = (
                f"rank {target} is above the {budget.max_lora_rank} the hardware budget "
                f"allows"
            )
        elif target * PARAMS_PER_RANK > budget.max_total_adapter_params:
            proposal.refused = (
                f"rank {target} would need about {target * PARAMS_PER_RANK / 1e6:.0f}M "
                f"adapter parameters, over the "
                f"{budget.max_total_adapter_params / 1e6:.0f}M budget"
            )
        return proposal

    if kind == "prune_module":
        proposal = Proposal(
            kind=kind,
            subject=trigger.subject,
            rationale=trigger.rationale,
            evidence=trigger.evidence,
            needs_approval=needs_approval,
            cost={"frees_adapter_slots": 1, "versions": trigger.evidence.get("versions", 0)},
            payload={"module": trigger.subject},
        )
        if not allowed:
            proposal.refused = "expansion.allow_pruning is off"
        elif registry.get(trigger.subject).always_on:
            proposal.refused = "an always-on module is not a candidate for archiving"
        return proposal

    proposal = Proposal(
        kind="merge_to_base",
        subject=trigger.subject,
        rationale=trigger.rationale,
        evidence=trigger.evidence,
        # Always, whatever the config says: this one rewrites the base checkpoint.
        needs_approval=True,
        cost={
            "rewrites": "the base checkpoint",
            "snapshot_includes_base": True,
            "frees_adapter_slots": 1,
            "invalidates_other_adapters": (
                "every other adapter becomes a delta from weights that no longer "
                "exist; they are stamped stale and skipped until retrained"
            ),
        },
        payload={"module": trigger.subject},
    )
    if not allowed:
        proposal.refused = "expansion.allow_base_merge is off"
    return proposal


def _module_name(word: str, registry: Registry) -> str:
    base = "".join(c if c.isalnum() else "-" for c in word.lower()).strip("-") or "topic"
    if base not in registry:
        return base
    for suffix in range(2, 20):
        candidate = f"{base}-{suffix}"
        if candidate not in registry:
            return candidate
    return f"{base}-new"


def _description_for(trigger: Trigger) -> str:
    """A description the router can match against, built from what was actually asked.

    It is deliberately a statement of observed vocabulary rather than a generated
    summary: the router scores against this text, so inventing prose about a subject
    the system does not yet have a module for would be routing on a guess.
    """
    keywords = trigger.evidence.get("keywords") or [trigger.subject]
    examples = trigger.evidence.get("examples") or []
    text = (
        f"Requests about {trigger.subject}, which no existing module covered. "
        f"Observed vocabulary: {', '.join(keywords[:10])}."
    )
    if examples:
        first = examples[0].strip().replace("\n", " ")[:160]
        text += f' For example: "{first}".'
    return text
