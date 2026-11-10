"""SQLite access layer.

One file, WAL mode, so the daemon and its child processes can all write. Every
schema change is an append-only migration; the DB is never recreated in place.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gustav.store.migrations import MIGRATIONS


class Database:
    """A connection to the Gustav store.

    Safe to share across threads (one lock, ``check_same_thread=False``) and safe to
    open independently from several processes (WAL + a busy timeout).
    """

    def __init__(self, path: Path | str, *, timeout: float = 30.0, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=timeout,
            check_same_thread=False,
            isolation_level=None,  # explicit transactions
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()

    def _configure(self) -> None:
        cur = self._conn.cursor()
        if str(self.path) != ":memory:":
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=FULL")  # survive kill -9 / power loss
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    # --- schema ---------------------------------------------------------
    def migrate(self) -> int:
        """Apply pending migrations. Returns the resulting schema version."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version INTEGER PRIMARY KEY,"
                " name TEXT NOT NULL,"
                " applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
            applied = {row[0] for row in cur.execute("SELECT version FROM schema_migrations")}
            for version, name, statements in MIGRATIONS:
                if version in applied:
                    continue
                cur.execute("BEGIN")
                try:
                    for stmt in statements:
                        cur.execute(stmt)
                    cur.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                        (version, name),
                    )
                    cur.execute("COMMIT")
                except Exception:
                    cur.execute("ROLLBACK")
                    raise
            version = cur.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()[0]
            cur.close()
            return int(version)

    # --- primitives -----------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executemany(sql, seq)

    def insert(self, table: str, values: dict[str, Any]) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        params = [_encode(v) for v in values.values()]
        with self._lock:
            cur = self._conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", params)
            return int(cur.lastrowid or 0)

    def query(
        self, sql: str, params: Sequence[Any] | dict[str, Any] = ()
    ) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(
        self, sql: str, params: Sequence[Any] | dict[str, Any] = ()
    ) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = (), default: Any = None):
        row = self.query_one(sql, params)
        return row[0] if row is not None else default

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # --- key/value ------------------------------------------------------
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM meta WHERE key = ?", (key,))
        if row is None:
            return default
        return json.loads(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO meta (key, value, updated_at) VALUES (?, ?, datetime('now'))"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, json.dumps(value)),
        )

    def delete_meta(self, key: str) -> None:
        self.execute("DELETE FROM meta WHERE key = ?", (key,))

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _encode(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    return value


def open_db(path: Path | str) -> Database:
    """Open and migrate in one step."""
    db = Database(path)
    db.migrate()
    return db


def row_to_dict(row: sqlite3.Row | None, json_fields: Sequence[str] = ()) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for field in json_fields:
        raw = out.get(field)
        if isinstance(raw, str) and raw:
            try:
                out[field] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return out
