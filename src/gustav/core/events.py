"""The event log and telemetry sink.

Both are plain SQLite tables so `ai status`, the HTML report, and tests can all read
the same history the daemon wrote.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from gustav.store.db import Database, row_to_dict


class Events:
    def __init__(self, db: Database, logger: logging.Logger | None = None):
        self.db = db
        self.log = logger or logging.getLogger("gustav.events")

    def emit(
        self,
        kind: str,
        message: str = "",
        *,
        level: str = "info",
        **data: Any,
    ) -> int:
        payload = json.dumps(data) if data else None
        event_id = self.db.insert(
            "events",
            {"kind": kind, "message": message, "level": level, "data": payload},
        )
        line = f"[{kind}] {message}" if message else f"[{kind}]"
        if data:
            line = f"{line} {payload}"
        self.log.log(getattr(logging, level.upper(), logging.INFO), line)
        return event_id

    def recent(self, limit: int = 50, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self.db.query(
                "SELECT * FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit)
            )
        else:
            rows = self.db.query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        return [row_to_dict(r, ("data",)) for r in rows]  # type: ignore[misc]


class Telemetry:
    def __init__(self, db: Database):
        self.db = db

    def record(self, metric: str, value: float, **tags: Any) -> None:
        self.db.insert(
            "telemetry",
            {"metric": metric, "value": float(value), "tags": json.dumps(tags) if tags else None},
        )

    def latest(self, metric: str) -> float | None:
        row = self.db.query_one(
            "SELECT value FROM telemetry WHERE metric = ? ORDER BY id DESC LIMIT 1", (metric,)
        )
        return float(row["value"]) if row else None
