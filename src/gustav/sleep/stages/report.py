"""Write the night's report.

Last step, and the only one that may not fail quietly: if the night did something the
user should know about, the report is how they find out. It is regenerated from SQLite
rather than from the other steps' return values, so `ai report <id>` can rebuild the
report of a run that crashed three steps in.
"""

from __future__ import annotations

from typing import Any

from gustav.report.html import write_report
from gustav.sleep.stages._base import StepEnv, sleep_step


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    if env.dry_run:
        return {"dry_run": True, "would_write": str(env.paths.reports / f"run-{env.run_id:06d}.html")}
    path = write_report(
        env.db, env.run_id, env.paths.reports, instance=env.config.instance.name
    )
    env.db.set_meta("sleep.last_report", str(path))
    env.events.emit(
        "sleep.report", f"report written to {path}", run_id=env.run_id, path=str(path)
    )
    return {"path": str(path), "bytes": path.stat().st_size}
