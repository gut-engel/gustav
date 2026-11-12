"""One line per sleep step, in the step's own units.

Shared by the HTML report and `ai runs <id>`, because a step's result dict is only
meaningful if something knows which of its keys matter — and that knowledge should not
exist twice and drift.
"""

from __future__ import annotations

from typing import Any

#: Keys worth showing per step, in order. A key whose value is zero or empty is left
#: out: "fetched 0, cached 0, failed 0" is noise, "fetched 3" is the news.
INTERESTING: dict[str, tuple[str, ...]] = {
    "snapshot": ("snapshot", "size_mb"),
    "consolidation": ("exchanges", "examples", "facts_new", "corrections", "endorsements",
                      "rewrite_tasks", "implicit_pending"),
    "reflect": ("user_failures", "user_successes"),
    "fetch_docs": ("fetched", "cached", "failed", "refused", "unlicensed"),
    "study": ("chunks", "examples", "tasks_admitted", "rejected_ungrounded"),
    "practice": ("passed", "attempted", "examples"),
    "dreaming": ("solved", "rewritten", "rejected", "samples"),
    "training": ("trained", "skipped"),
    "retest": ("corrections_now_right", "corrections_retested", "tasks_now_passing"),
    "report": ("bytes",),
}


def step_summary(name: str, result: dict[str, Any] | None) -> str:
    """A short, human-readable account of what one step did."""
    result = result or {}
    if not result:
        return ""
    if result.get("placeholder"):
        return "simulated (this step lands in a later stage)"

    if name == "gate":
        bits = []
        if result.get("promoted"):
            bits.append("promoted " + ", ".join(result["promoted"]))
        if result.get("rejected"):
            bits.append("rejected " + ", ".join(result["rejected"]))
        if result.get("skipped"):
            bits.append("not promoted: " + ", ".join(result["skipped"]))
        if result.get("reason"):
            bits.append(str(result["reason"]))
        return "; ".join(bits) or "nothing to gate"

    if name == "expansion":
        bits = []
        for item in result.get("applied") or []:
            bits.append(f"{item.get('kind')} {item.get('subject')}")
        for item in result.get("pending") or []:
            bits.append(f"{item.get('kind')} {item.get('subject')} needs approval")
        if result.get("refused"):
            bits.append(f"{len(result['refused'])} refused")
        if not bits and result.get("reason"):
            return str(result["reason"])
        return "; ".join(bits) or "nothing to grow"

    if name == "speed":
        bits = []
        router = result.get("router") or {}
        if router.get("installed"):
            bits.append("router refitted")
        elif router.get("reason"):
            bits.append(f"router kept ({router['reason']})")
        optimisation = result.get("optimisation") or {}
        if optimisation.get("kept"):
            bits.append(
                "kept " + ", ".join(f"{k}={v}" for k, v in optimisation["kept"].items())
            )
        elif optimisation.get("trials"):
            bits.append(f"{len(optimisation['trials'])} trial(s), nothing kept")
        draft = result.get("draft") or {}
        if draft.get("acceptance") is not None:
            bits.append(f"speculation accepted {draft['acceptance']:.0%}")
        return "; ".join(bits) or str(result.get("reason") or "")

    if name == "reflect" and result.get("topics"):
        topics = ", ".join(
            f"{t.get('module')}/{t.get('topic')}" for t in result["topics"][:6]
        )
        return f"topics: {topics}"

    parts = [
        f"{key.replace('_', ' ')} {result[key]}"
        for key in INTERESTING.get(name, ())
        if result.get(key) not in (None, 0, "", [], {})
    ]
    if parts:
        return ", ".join(parts)
    if result.get("reason"):
        return str(result["reason"])
    if result.get("dry_run"):
        return "dry run"
    return ""
