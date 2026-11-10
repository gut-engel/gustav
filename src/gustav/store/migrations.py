"""Append-only schema migrations.

Never edit a migration that has shipped; add a new one. ``MIGRATIONS`` is a list of
``(version, name, [sql, ...])``.
"""

from __future__ import annotations

MIGRATION_0001 = [
    # --- daemon bookkeeping -------------------------------------------------
    """
    CREATE TABLE meta (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE events (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        ts      TEXT NOT NULL DEFAULT (datetime('now')),
        level   TEXT NOT NULL DEFAULT 'info',
        kind    TEXT NOT NULL,
        message TEXT NOT NULL DEFAULT '',
        data    TEXT
    )
    """,
    "CREATE INDEX idx_events_ts ON events (ts DESC)",
    "CREATE INDEX idx_events_kind ON events (kind, ts DESC)",
    """
    CREATE TABLE state_history (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL DEFAULT (datetime('now')),
        from_state TEXT,
        to_state   TEXT NOT NULL,
        reason     TEXT
    )
    """,
    # --- wake phase ---------------------------------------------------------
    """
    CREATE TABLE sessions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        external_id      TEXT UNIQUE,
        source           TEXT NOT NULL DEFAULT 'cli',
        label            TEXT,
        started_at       TEXT NOT NULL DEFAULT (datetime('now')),
        last_activity_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE turns (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        ts               TEXT NOT NULL DEFAULT (datetime('now')),
        role             TEXT NOT NULL,
        content          TEXT NOT NULL,
        -- which weights produced this answer, so before/after a sleep is comparable
        adapter_versions TEXT,
        base_model       TEXT,
        prompt_tokens    INTEGER,
        completion_tokens INTEGER,
        latency_ms       INTEGER,
        tokens_per_second REAL,
        -- 0 means: never train on this (/forget, detected secrets, ...)
        trainable        INTEGER NOT NULL DEFAULT 1,
        redacted         INTEGER NOT NULL DEFAULT 0,
        exclude_reason   TEXT
    )
    """,
    "CREATE INDEX idx_turns_session ON turns (session_id, id)",
    "CREATE INDEX idx_turns_ts ON turns (ts DESC)",
    """
    CREATE TABLE feedback (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL DEFAULT (datetime('now')),
        turn_id    INTEGER REFERENCES turns(id) ON DELETE CASCADE,
        kind       TEXT NOT NULL,           -- good | bad | correct | implicit
        payload    TEXT,                    -- e.g. the corrected answer
        source     TEXT NOT NULL DEFAULT 'user',
        reviewed   INTEGER NOT NULL DEFAULT 0,
        consumed_run_id INTEGER
    )
    """,
    "CREATE INDEX idx_feedback_open ON feedback (consumed_run_id, kind)",
    # --- pending work (a sleep trigger) ------------------------------------
    """
    CREATE TABLE pending_work (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        kind       TEXT NOT NULL,
        weight     INTEGER NOT NULL DEFAULT 1,
        ref        TEXT,
        consumed_run_id INTEGER
    )
    """,
    "CREATE INDEX idx_pending_open ON pending_work (consumed_run_id)",
    # --- sleep journal ------------------------------------------------------
    """
    CREATE TABLE sleep_runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        kind        TEXT NOT NULL DEFAULT 'sleep',
        trigger     TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'running',
                    -- running | completed | failed | interrupted | paused | aborted
        dry_run     INTEGER NOT NULL DEFAULT 0,
        started_at  TEXT NOT NULL DEFAULT (datetime('now')),
        finished_at TEXT,
        run_dir     TEXT,
        pid         INTEGER,
        reason      TEXT,
        summary     TEXT
    )
    """,
    "CREATE INDEX idx_sleep_runs_status ON sleep_runs (status, id DESC)",
    """
    CREATE TABLE sleep_steps (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES sleep_runs(id) ON DELETE CASCADE,
        name        TEXT NOT NULL,
        position    INTEGER NOT NULL,
        status      TEXT NOT NULL DEFAULT 'pending',
                    -- pending | running | done | failed | skipped | preempted
        attempts    INTEGER NOT NULL DEFAULT 0,
        started_at  TEXT,
        finished_at TEXT,
        heartbeat_at TEXT,
        checkpoint  TEXT,
        result      TEXT,
        error       TEXT,
        on_crash    TEXT NOT NULL DEFAULT 'resume',
        UNIQUE (run_id, name)
    )
    """,
    "CREATE INDEX idx_sleep_steps_run ON sleep_steps (run_id, position)",
    # --- approvals ----------------------------------------------------------
    """
    CREATE TABLE approvals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        kind        TEXT NOT NULL,     -- base_change | expansion | prune | ...
        subject     TEXT NOT NULL,
        rationale   TEXT,
        payload     TEXT,
        snapshot_id TEXT,
        status      TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | expired
        decided_at  TEXT,
        note        TEXT
    )
    """,
    "CREATE INDEX idx_approvals_status ON approvals (status, id DESC)",
    # --- telemetry ----------------------------------------------------------
    """
    CREATE TABLE telemetry (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        ts     TEXT NOT NULL DEFAULT (datetime('now')),
        metric TEXT NOT NULL,
        value  REAL NOT NULL,
        tags   TEXT
    )
    """,
    "CREATE INDEX idx_telemetry_metric ON telemetry (metric, ts DESC)",
]

MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "initial_daemon_schema", MIGRATION_0001),
]
