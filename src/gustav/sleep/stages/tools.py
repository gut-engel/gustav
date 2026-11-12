"""Tools: write what was asked for, and let a human read it.

Section 11. The night writes tools from two sources, and nothing else:

* **what you asked for** — `ai tools request "convert units"` leaves a row the night
  picks up, which is the honest way round: you know what you need a tool for, and the
  model does not;
* **what it keeps failing at** — a practice task that failed every time it was tried,
  where the failure is the kind a tool fixes (arithmetic, dates, parsing) rather than
  the kind more training fixes.

It does not write tools speculatively. A tool is source code that will run on this
machine, and a queue of plausible-looking tools nobody asked for is a queue nobody
reads — at which point the review is theatre.

Everything written here lands as `proposed`. The step cannot approve anything; neither
can the daemon. Only `ai tools approve` can, and only after the source has been shown.
"""

from __future__ import annotations

from typing import Any

from gustav.sleep.stages._base import StepEnv, sleep_step
from gustav.tools.author import write_tool
from gustav.tools.runner import ToolRunner
from gustav.tools.store import ToolError, ToolStore

#: Words in a failed task that suggest a tool would help more than more training would.
COMPUTATIONAL = (
    "calculate", "compute", "convert", "parse", "timezone", "date", "arithmetic",
    "round", "percentage", "currency", "unit", "checksum", "hash", "encode", "decode",
    "regex", "sort", "statistics", "average", "median", "random",
)


def _needs_from_failures(env: StepEnv, limit: int) -> list[str]:
    """Tasks that failed every attempt and look like a tool would fix them."""
    rows = env.db.query(
        "SELECT t.prompt, t.attempts, t.passes FROM tasks t"
        " WHERE t.attempts >= 2 AND t.passes = 0 AND t.status != 'retired'"
        " ORDER BY t.attempts DESC LIMIT 50"
    )
    needs = []
    for row in rows:
        prompt = (row["prompt"] or "").lower()
        if any(word in prompt for word in COMPUTATIONAL):
            needs.append(
                f"a tool that would help with this, which the model keeps failing: "
                f"{row['prompt'][:300]}"
            )
        if len(needs) >= limit:
            break
    return needs


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    cfg = env.config.tools
    store = ToolStore(env.db, env.paths.tools, max_tools=cfg.max_tools)
    result: dict[str, Any] = {
        "requests": 0,
        "written": [],
        "refused": [],
        "pending_review": len(store.pending()),
        "approved": len(store.approved()),
    }
    if not cfg.allow_authoring:
        result["reason"] = "tools.allow_authoring is false"
        return result

    requests = store.open_requests(limit=cfg.max_new_per_run)
    needs: list[tuple[int | None, str]] = [(int(r["id"]), r["text"]) for r in requests]
    if len(needs) < cfg.max_new_per_run:
        needs += [
            (None, text)
            for text in _needs_from_failures(env, cfg.max_new_per_run - len(needs))
        ]
    result["requests"] = len(needs)
    if not needs:
        result["reason"] = "nothing asked for a tool, and nothing failed like a tool would fix"
        return result

    model = env.model()
    if not model.available:
        result["reason"] = model.error or "no model available"
        return result
    if env.dry_run:
        result["dry_run"] = True
        result["would_write"] = [text[:120] for _id, text in needs]
        return result

    runner = ToolRunner(workspace_root=env.paths.tmp)
    for _index, (request_id, text) in env.each(needs, label="tool", checkpoint_key="need_index"):
        env.progress(f"writing a tool for: {text[:60]}")
        authored = write_tool(model, text, cfg, runner=runner)
        if not authored.ok or authored.spec is None or authored.checks is None:
            result["refused"].append({"request": text[:120], "reason": authored.reason[:300]})
            if request_id is not None:
                store.close_request(
                    request_id, status="refused", detail=authored.reason[:400]
                )
            env.log(f"no tool for {text[:60]!r}: {authored.reason[:200]}", level="warning")
            continue
        try:
            version = store.propose(
                authored.spec,
                authored.checks,
                run_id=env.run_id,
                request_id=request_id,
            )
        except ToolError as exc:
            result["refused"].append({"request": text[:120], "reason": str(exc)[:300]})
            if request_id is not None:
                store.close_request(request_id, status="refused", detail=str(exc)[:400])
            continue

        result["written"].append(
            {
                "name": version.name,
                "version": version.version,
                "capabilities": list(version.capabilities),
                "attempts": authored.attempts,
                "path": str(version.path),
            }
        )
        env.events.emit(
            "tool.proposed",
            f"wrote {version.name}@{version.version} ({', '.join(version.capabilities)}); "
            f"it does nothing until you approve it: ai tools show {version.name}",
            level="warning",
            run_id=env.run_id,
            tool=version.name,
            version=version.version,
        )

    result["pending_review"] = len(store.pending())
    env.write_json("tools.json", result)
    return result
