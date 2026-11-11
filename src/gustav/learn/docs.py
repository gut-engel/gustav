"""The document cache, as a table.

A row per URL: when it was fetched, which topic asked for it, where the extracted text
landed, and which run has already studied it. That last column is what stops a night
re-deriving the same hundred study examples from the same page.
"""

from __future__ import annotations

from typing import Any

from gustav.store.db import Database


class DocumentStore:
    def __init__(self, db: Database):
        self.db = db

    def record(
        self,
        *,
        url: str,
        topic: str | None,
        title: str,
        sha256: str,
        size: int,
        path: str,
        host: str = "",
        content_type: str = "",
        license: str | None = None,
        status: str = "fresh",
        run_id: int | None = None,
        error: str | None = None,
    ) -> int:
        existing = self.db.query_one("SELECT id FROM documents WHERE url = ?", (url,))
        values = {
            "topic": topic,
            "title": title,
            "sha256": sha256,
            "bytes": int(size),
            "path": path,
            "host": host,
            "content_type": content_type,
            "license": license,
            "status": status,
            "run_id": run_id,
            "error": error,
        }
        if existing is None:
            return self.db.insert("documents", {"url": url, **values})
        sets = ", ".join(f"{k} = ?" for k in values)
        self.db.execute(
            f"UPDATE documents SET {sets}, fetched_at = datetime('now') WHERE id = ?",
            [*values.values(), int(existing["id"])],
        )
        return int(existing["id"])

    def age_hours(self, url: str) -> float | None:
        row = self.db.query_one(
            "SELECT (julianday('now') - julianday(fetched_at)) * 24 AS hours,"
            " status FROM documents WHERE url = ?",
            (url,),
        )
        if row is None or row["status"] == "failed":
            return None
        return float(row["hours"] or 0.0)

    def unstudied(self, *, limit: int = 20, run_id: int | None = None) -> list[dict[str, Any]]:
        """Documents this run fetched (or earlier ones never studied), oldest first."""
        sql = [
            "SELECT * FROM documents WHERE status != 'failed' AND path IS NOT NULL",
            "AND studied_run_id IS NULL",
        ]
        params: list[Any] = []
        if run_id is not None:
            sql.append("AND (run_id = ? OR run_id IS NULL)")
            params.append(run_id)
        sql.append("ORDER BY id LIMIT ?")
        params.append(int(limit))
        return [dict(r) for r in self.db.query(" ".join(sql), params)]

    def mark_studied(self, document_id: int, run_id: int) -> None:
        self.db.execute(
            "UPDATE documents SET studied_run_id = ? WHERE id = ?", (run_id, document_id)
        )

    def counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM documents GROUP BY status")
        return {r["status"]: int(r["n"]) for r in rows}
