"""The morning report.

One self-contained HTML file per sleep run: no external stylesheet, no fonts, no
JavaScript, so it opens from the filesystem in ten years. It is written for the question
you actually have at breakfast — *what did it change, and should I trust it* — so the
decisions come first and the telemetry last.

Everything in it is read back out of SQLite rather than passed in from the steps, which
means a report can be regenerated for any past run (`ai report <id>`), including one
that crashed.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from gustav.report.summary import step_summary
from gustav.store.db import Database

STYLE = """
:root {
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1b1a18; --muted: #6b6760;
  --line: #e4e0d9; --accent: #7a5c2e; --good: #2f6b43; --bad: #9b3030; --warn: #8a6d1f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #15161a; --panel: #1c1e23; --ink: #e9e7e2; --muted: #9b968d;
    --line: #2c2f36; --accent: #d7b377; --good: #7bc292; --bad: #e08a8a; --warn: #d9bd6a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1rem 4rem; background: var(--bg); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .2rem; }
h2 { font-size: 1.1rem; margin: 2.2rem 0 .6rem; letter-spacing: .01em; }
.sub { color: var(--muted); margin: 0 0 1.6rem; }
.cards { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: .6rem;
  padding: .85rem .95rem;
}
.card .n { font-size: 1.5rem; font-variant-numeric: tabular-nums; }
.card .l { color: var(--muted); font-size: .8rem; text-transform: uppercase;
           letter-spacing: .06em; }
table { width: 100%; border-collapse: collapse; background: var(--panel);
        border: 1px solid var(--line); border-radius: .6rem; overflow: hidden; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
     color: var(--muted); font-weight: 600; }
tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em; }
pre { background: var(--panel); border: 1px solid var(--line); border-radius: .5rem;
      padding: .7rem; overflow-x: auto; }
.pill { display: inline-block; padding: .05rem .45rem; border-radius: 1rem;
        font-size: .75rem; border: 1px solid var(--line); color: var(--muted); }
.good { color: var(--good); } .bad { color: var(--bad); } .warn { color: var(--warn); }
.muted { color: var(--muted); }
ul { margin: .3rem 0 0; padding-left: 1.2rem; }
li { margin: .15rem 0; }
footer { color: var(--muted); font-size: .8rem; margin-top: 3rem;
         border-top: 1px solid var(--line); padding-top: .8rem; }
"""

_STATUS_CLASS = {
    "completed": "good", "done": "good", "promoted": "good", "pass": "good",
    "failed": "bad", "fail": "bad", "rejected": "bad", "aborted": "bad",
    "paused": "warn", "preempted": "warn", "interrupted": "warn", "skipped": "muted",
    "skip": "muted", "pending": "muted", "held": "warn",
}


def _pill(text: str) -> str:
    cls = _STATUS_CLASS.get(str(text).lower(), "muted")
    return f'<span class="pill {cls}">{escape(str(text))}</span>'


def _card(label: str, value: Any, note: str = "") -> str:
    extra = f'<div class="l">{escape(note)}</div>' if note else ""
    return (
        f'<div class="card"><div class="n">{escape(str(value))}</div>'
        f'<div class="l">{escape(label)}</div>{extra}</div>'
    )


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _duration(started: str | None, finished: str | None) -> str:
    if not started or not finished:
        return "—"
    try:
        a = datetime.fromisoformat(started)
        b = datetime.fromisoformat(finished)
    except ValueError:
        return "—"
    seconds = max((b - a).total_seconds(), 0)
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"


def _step_summary(name: str, result: dict[str, Any]) -> str:
    """The shared one-liner, as HTML. Muted when the step has nothing to report."""
    text = step_summary(name, result)
    if not text:
        return '<span class="muted">—</span>'
    if result.get("placeholder") or result.get("reason") == text:
        return f'<span class="muted">{escape(text)}</span>'
    return escape(text)


def render_run(db: Database, run_id: int, *, instance: str = "gustav") -> str:
    run = db.query_one("SELECT * FROM sleep_runs WHERE id = ?", (run_id,))
    if run is None:
        raise KeyError(f"there is no sleep run {run_id}")
    steps = db.query(
        "SELECT * FROM sleep_steps WHERE run_id = ? ORDER BY position", (run_id,)
    )
    results = {row["name"]: _loads(row["result"]) for row in steps}

    examples = db.query(
        "SELECT kind, COUNT(*) AS n, SUM(split = 'heldout') AS held"
        " FROM training_examples WHERE run_id = ? GROUP BY kind ORDER BY n DESC",
        (run_id,),
    )
    facts = db.query(
        "SELECT text, subject, confidence FROM facts WHERE run_id = ?"
        " ORDER BY confidence DESC LIMIT 25",
        (run_id,),
    )
    trainings = db.query(
        "SELECT * FROM training_runs WHERE run_id = ? ORDER BY module", (run_id,)
    )
    gates = db.query(
        "SELECT * FROM gate_results WHERE run_id = ? ORDER BY module, suite", (run_id,)
    )
    skills = db.query(
        "SELECT * FROM skills WHERE trials > 0 ORDER BY score ASC, demand DESC LIMIT 12"
    )
    events = db.query(
        "SELECT ts, level, kind, message FROM events WHERE kind LIKE 'sleep%'"
        " AND json_extract(data, '$.run_id') = ? ORDER BY id DESC LIMIT 40",
        (run_id,),
    )
    docs = db.query(
        "SELECT url, title, topic, status, license FROM documents WHERE run_id = ?"
        " ORDER BY id LIMIT 20",
        (run_id,),
    )

    total_examples = sum(int(r["n"]) for r in examples)
    promoted = [r["module"] for r in trainings if r["status"] == "promoted"]
    practice = results.get("practice", {})
    gate_result = results.get("gate", {})

    out: list[str] = []
    out.append(
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>{escape(instance)} — sleep run {run_id}</title>"
        f"<style>{STYLE}</style></head><body><main>"
    )
    out.append(f"<h1>Sleep run {run_id} {_pill(run['status'])}</h1>")
    out.append(
        f"<p class=sub>{escape(run['trigger'])} · started {escape(run['started_at'] or '—')}"
        f" · {_duration(run['started_at'], run['finished_at'])}"
        + (" · dry run" if run["dry_run"] else "")
        + (f" · {escape(str(run['reason']))}" if run["reason"] else "")
        + "</p>"
    )

    out.append('<div class=cards>')
    out.append(_card("examples", total_examples, "learned from tonight"))
    out.append(_card("new facts", len(facts)))
    out.append(
        _card(
            "tasks passed",
            f"{practice.get('passed', 0)}/{practice.get('attempted', 0)}",
            "verified by tests",
        )
    )
    out.append(_card("documents", len(docs)))
    out.append(_card("promoted", len(promoted) or "none", ", ".join(promoted)))
    out.append("</div>")

    # --- what changed ---------------------------------------------------
    out.append("<h2>What changed</h2>")
    if trainings:
        out.append(
            "<table><tr><th>module</th><th>version</th><th>status</th>"
            "<th class=num>examples</th><th class=num>steps</th><th>loss</th>"
            "<th>why</th></tr>"
        )
        for row in trainings:
            detail = _loads(row["detail"])
            loss = (
                f"{row['loss_start']:.3f} → {row['loss_end']:.3f}"
                if row["loss_start"] is not None and row["loss_end"] is not None
                else "—"
            )
            out.append(
                f"<tr><td>{escape(row['module'])}</td>"
                f"<td><code>{escape(row['version'] or '—')}</code></td>"
                f"<td>{_pill(row['status'])}</td>"
                f"<td class=num>{row['examples']}</td>"
                f"<td class=num>{row['steps']}</td><td>{escape(loss)}</td>"
                f"<td class=muted>{escape(str(detail.get('gate', '')))[:160]}</td></tr>"
            )
        out.append("</table>")
    else:
        reason = gate_result.get("reason") or results.get("training", {}).get(
            "reason", "nothing was trained"
        )
        out.append(f"<p class=muted>{escape(str(reason))}</p>")

    if gates:
        out.append("<h2>The gate</h2><table><tr><th>module</th><th>suite</th>"
                   "<th class=num>items</th><th class=num>before</th><th class=num>after</th>"
                   "<th class=num>delta</th><th>verdict</th></tr>")
        for row in gates:
            delta = row["delta"]
            cls = "good" if (delta or 0) > 0 else ("bad" if (delta or 0) < 0 else "muted")
            out.append(
                f"<tr><td>{escape(row['module'])}</td><td>{escape(row['suite'])}</td>"
                f"<td class=num>{row['items']}</td>"
                f"<td class=num>{_fmt(row['baseline_score'])}</td>"
                f"<td class=num>{_fmt(row['candidate_score'])}</td>"
                f"<td class='num {cls}'>{_fmt(delta, signed=True)}</td>"
                f"<td>{_pill(row['verdict'])}</td></tr>"
            )
        out.append("</table>")

    # --- the night, step by step ---------------------------------------
    out.append("<h2>The night, step by step</h2>")
    out.append(
        "<table><tr><th>step</th><th>status</th><th class=num>tries</th>"
        "<th class=num>took</th><th>what it did</th></tr>"
    )
    for row in steps:
        out.append(
            f"<tr><td><code>{escape(row['name'])}</code></td>"
            f"<td>{_pill(row['status'])}</td>"
            f"<td class=num>{row['attempts']}</td>"
            f"<td class=num>{_duration(row['started_at'], row['finished_at'])}</td>"
            f"<td>{_step_summary(row['name'], results.get(row['name'], {}))}"
            + (
                f"<br><span class=bad>{escape(str(row['error'])[:200])}</span>"
                if row["error"]
                else ""
            )
            + "</td></tr>"
        )
    out.append("</table>")

    # --- material -------------------------------------------------------
    if examples:
        out.append("<h2>What it learned from</h2><table><tr><th>kind</th>"
                   "<th class=num>examples</th><th class=num>held out</th></tr>")
        for row in examples:
            out.append(
                f"<tr><td>{escape(row['kind'])}</td><td class=num>{row['n']}</td>"
                f"<td class=num>{int(row['held'] or 0)}</td></tr>"
            )
        out.append("</table>")

    if facts:
        out.append("<h2>Facts it now holds</h2><ul>")
        for row in facts:
            subject = f" <span class=pill>{escape(row['subject'])}</span>" if row["subject"] else ""
            out.append(
                f"<li>{escape(row['text'])}{subject} "
                f"<span class=muted>{float(row['confidence']):.2f}</span></li>"
            )
        out.append("</ul>")

    if docs:
        out.append("<h2>What it read</h2><ul>")
        for row in docs:
            label = row["title"] or row["url"]
            licence = row["license"] or "licence not recorded"
            cls = "pill" if row["license"] else "pill warn"
            out.append(
                f"<li>{escape(label)} <span class=muted>({escape(row['topic'] or '—')}, "
                f"{escape(row['status'])})</span> "
                f"<span class='{cls}'>{escape(licence)}</span><br>"
                f"<code class=muted>{escape(row['url'])}</code></li>"
            )
        out.append("</ul>")

    if skills:
        out.append("<h2>Where it still struggles</h2><table><tr><th>module</th><th>topic</th>"
                   "<th class=num>score</th><th class=num>trials</th>"
                   "<th class=num>demand</th></tr>")
        for row in skills:
            out.append(
                f"<tr><td>{escape(row['module'])}</td><td>{escape(row['topic'])}</td>"
                f"<td class=num>{float(row['score']):.2f}</td>"
                f"<td class=num>{row['trials']}</td>"
                f"<td class=num>{float(row['demand']):.2f}</td></tr>"
            )
        out.append("</table>")

    gaps = results.get("reflect", {}).get("gaps") or []
    if gaps:
        out.append(
            f"<h2>Requests nothing covered</h2><p class=muted>{len(gaps)} request(s) scored "
            f"below the router's threshold for every module. That is the measured trigger "
            f"for growing a new module — a decision that stays yours.</p>"
        )

    if events:
        out.append("<h2>Log</h2><table><tr><th>when</th><th>event</th><th>message</th></tr>")
        for row in reversed(events):
            cls = {"warning": "warn", "error": "bad"}.get(row["level"], "")
            out.append(
                f"<tr><td class=muted>{escape(row['ts'])}</td>"
                f"<td><code>{escape(row['kind'])}</code></td>"
                f"<td class='{cls}'>{escape(row['message'])}</td></tr>"
            )
        out.append("</table>")

    out.append(
        f"<footer>{escape(instance)} · generated "
        f"{datetime.now().isoformat(timespec='seconds')} · "
        f"run directory <code>{escape(str(run['run_dir'] or '—'))}</code></footer>"
    )
    out.append("</main></body></html>")
    return "".join(out)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    return f"{number:+.2f}" if signed else f"{number:.2f}"


def write_report(
    db: Database, run_id: int, directory: Path, *, instance: str = "gustav"
) -> Path:
    """Write the run's report and update ``latest.html`` to point at it."""
    directory.mkdir(parents=True, exist_ok=True)
    html = render_run(db, run_id, instance=instance)
    path = directory / f"run-{run_id:06d}.html"
    path.write_text(html, encoding="utf-8")
    latest = directory / "latest.html"
    tmp = directory / ".latest.html.part"
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(latest)
    return path
