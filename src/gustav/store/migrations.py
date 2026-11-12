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

MIGRATION_0002 = [
    # Section 9: every operation that touches weights records the snapshot it took
    # first. An entry still 'in_progress' at startup is an interrupted change, which
    # is exactly what crash recovery needs to find and undo.
    """
    CREATE TABLE weight_changes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at  TEXT NOT NULL DEFAULT (datetime('now')),
        finished_at TEXT,
        operation   TEXT NOT NULL,
        snapshot_id TEXT,
        status      TEXT NOT NULL DEFAULT 'in_progress',
                    -- in_progress | done | rolled_back | failed | refused
        run_id      INTEGER,
        step        TEXT,
        sources     TEXT,
        detail      TEXT
    )
    """,
    "CREATE INDEX idx_weight_changes_status ON weight_changes (status, id DESC)",
    "CREATE INDEX idx_weight_changes_step ON weight_changes (run_id, step)",
]

MIGRATION_0003 = [
    # --- what the day left behind -------------------------------------------
    # Durable statements about the user and their world. Consolidation writes them,
    # `ai facts` shows them, and the personal-memory module trains on them. A fact is
    # never edited in place: a newer one supersedes it, so the history stays readable.
    """
    CREATE TABLE facts (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
        module        TEXT NOT NULL DEFAULT 'personal-memory',
        subject       TEXT,
        text          TEXT NOT NULL,
        confidence    REAL NOT NULL DEFAULT 0.5,
        source        TEXT,
        source_turn_id INTEGER,
        run_id        INTEGER,
        status        TEXT NOT NULL DEFAULT 'active',  -- active | superseded | retracted
        superseded_by INTEGER,
        fingerprint   TEXT NOT NULL UNIQUE
    )
    """,
    "CREATE INDEX idx_facts_active ON facts (status, subject)",
    # Every candidate the night produced, whatever produced it. One table, because the
    # trainer's job is to mix sources in known proportions and it can only do that if
    # they are comparable: same columns, a weight, and a fingerprint so the same example
    # discovered twice is stored once.
    """
    CREATE TABLE training_examples (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        run_id        INTEGER,
        module        TEXT NOT NULL DEFAULT 'general',
        kind          TEXT NOT NULL,
                      -- correction | preference | fact | study | practice | dream
        prompt        TEXT NOT NULL,
        completion    TEXT NOT NULL,
        rejected      TEXT,                            -- the worse answer of a pair
        weight        REAL NOT NULL DEFAULT 1.0,
        source        TEXT,
        source_ref    TEXT,
        citation      TEXT,                            -- where a study example came from
        -- 1 only if something other than the model said it was right: the user, a test,
        -- a compiler. The trainer can be told to use nothing else.
        verified      INTEGER NOT NULL DEFAULT 0,
        split         TEXT NOT NULL DEFAULT 'train',   -- train | heldout
        status        TEXT NOT NULL DEFAULT 'ready',   -- ready | used | dropped
        trained_run_id INTEGER,
        fingerprint   TEXT NOT NULL UNIQUE
    )
    """,
    "CREATE INDEX idx_examples_ready ON training_examples (status, split, module)",
    "CREATE INDEX idx_examples_run ON training_examples (run_id, kind)",
    # --- study --------------------------------------------------------------
    """
    CREATE TABLE documents (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
        url          TEXT NOT NULL UNIQUE,
        host         TEXT,
        topic        TEXT,
        title        TEXT,
        sha256       TEXT,
        bytes        INTEGER NOT NULL DEFAULT 0,
        path         TEXT,                             -- relative to var/doc_cache
        content_type TEXT,
        -- What docs/sources.yaml declared for this source. Training on documentation
        -- is a licence question, and an unrecorded licence is one you cannot answer
        -- later, so a missing value is stored as NULL and warned about rather than
        -- quietly assumed to be fine.
        license      TEXT,
        status       TEXT NOT NULL DEFAULT 'fresh',    -- fresh | cached | failed | skipped
        run_id       INTEGER,
        error        TEXT,
        studied_run_id INTEGER
    )
    """,
    "CREATE INDEX idx_documents_topic ON documents (topic, fetched_at DESC)",
    # --- practice -----------------------------------------------------------
    # A task carries its own verifier, so "did it pass" is never the model's opinion.
    """
    CREATE TABLE tasks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        run_id      INTEGER,
        module      TEXT NOT NULL DEFAULT 'general',
        topic       TEXT,
        kind        TEXT NOT NULL,                     -- python | qa
        prompt      TEXT NOT NULL,
        verifier    TEXT NOT NULL,                     -- JSON: tests, or the answer
        difficulty  REAL NOT NULL DEFAULT 0.5,
        source      TEXT,
        source_ref  TEXT,
        document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
        status      TEXT NOT NULL DEFAULT 'open',      -- open | solved | failed | retired
        attempts    INTEGER NOT NULL DEFAULT 0,
        passes      INTEGER NOT NULL DEFAULT 0,
        fingerprint TEXT NOT NULL UNIQUE
    )
    """,
    "CREATE INDEX idx_tasks_open ON tasks (status, module, topic)",
    """
    CREATE TABLE task_attempts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        task_id     INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        run_id      INTEGER,
        step        TEXT,
        attempt     INTEGER NOT NULL DEFAULT 1,
        answer      TEXT,
        passed      INTEGER NOT NULL DEFAULT 0,
        detail      TEXT,
        duration_ms INTEGER
    )
    """,
    "CREATE INDEX idx_attempts_task ON task_attempts (task_id, id DESC)",
    # --- what it can and cannot do -----------------------------------------
    # The measured skill table. `reflect` ranks on it, `retest` updates it, and it is
    # what makes "it got worse at Rust" a number rather than an impression.
    """
    CREATE TABLE skills (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        module        TEXT NOT NULL,
        topic         TEXT NOT NULL,
        trials        INTEGER NOT NULL DEFAULT 0,
        passes        INTEGER NOT NULL DEFAULT 0,
        score         REAL NOT NULL DEFAULT 0.0,       -- smoothed pass rate
        demand        REAL NOT NULL DEFAULT 0.0,       -- how often it is actually asked
        first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
        last_seen_at  TEXT NOT NULL DEFAULT (datetime('now')),
        studied_at    TEXT,
        note          TEXT,
        UNIQUE (module, topic)
    )
    """,
    # --- the gate -----------------------------------------------------------
    # Held-out and regression suites. Items are never trained on: the dataset builder
    # filters on split, and benchmark items live here rather than in the example table
    # so there is no way to accidentally select them.
    """
    CREATE TABLE benchmark_items (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        suite       TEXT NOT NULL,                     -- heldout | regression
        module      TEXT NOT NULL DEFAULT 'general',
        topic       TEXT,
        kind        TEXT NOT NULL,                     -- qa | python
        prompt      TEXT NOT NULL,
        expected    TEXT,
        verifier    TEXT,
        run_id      INTEGER,
        retired     INTEGER NOT NULL DEFAULT 0,
        fingerprint TEXT NOT NULL UNIQUE
    )
    """,
    "CREATE INDEX idx_bench_suite ON benchmark_items (suite, module, retired)",
    """
    CREATE TABLE training_runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        finished_at TEXT,
        run_id      INTEGER,
        module      TEXT NOT NULL,
        version     TEXT,
        examples    INTEGER NOT NULL DEFAULT 0,
        replay      INTEGER NOT NULL DEFAULT 0,
        steps       INTEGER NOT NULL DEFAULT 0,
        loss_start  REAL,
        loss_end    REAL,
        seconds     REAL,
        status      TEXT NOT NULL DEFAULT 'running',
                    -- running | trained | promoted | rejected | failed | skipped
        detail      TEXT
    )
    """,
    "CREATE INDEX idx_training_runs_run ON training_runs (run_id, module)",
    """
    CREATE TABLE gate_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        run_id          INTEGER,
        module          TEXT NOT NULL,
        candidate       TEXT,
        baseline        TEXT,
        suite           TEXT NOT NULL,
        items           INTEGER NOT NULL DEFAULT 0,
        candidate_score REAL,
        baseline_score  REAL,
        delta           REAL,
        verdict         TEXT NOT NULL DEFAULT 'pending',  -- pass | fail | skip
        detail          TEXT
    )
    """,
    "CREATE INDEX idx_gate_run ON gate_results (run_id, module)",
]

MIGRATION_0004 = [
    # Section 8: every structural change the system proposes about itself, whether it
    # was applied, waved through, refused or reverted. The snapshot id is what makes
    # `ai expansion --revert` possible, and the evidence is what makes the proposal
    # reviewable instead of a thing that happened in the night.
    """
    CREATE TABLE expansion_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        decided_at  TEXT,
        run_id      INTEGER,
        kind        TEXT NOT NULL,
                    -- new_module | raise_rank | prune_module | merge_to_base
        subject     TEXT NOT NULL,
        rationale   TEXT,
        evidence    TEXT,
        cost        TEXT,
        payload     TEXT,
        status      TEXT NOT NULL DEFAULT 'proposed',
                    -- proposed | applied | pending | rejected | reverted | failed
        approval_id INTEGER,
        snapshot_id TEXT,
        detail      TEXT
    )
    """,
    "CREATE INDEX idx_expansion_status ON expansion_log (status, id DESC)",
    # Section 10: self-optimisation is only allowed to touch `tunable`, and only on a
    # measurement. One row per trial: what it changed, what it measured before and
    # after, and whether the change was kept.
    """
    CREATE TABLE optimisations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        run_id      INTEGER,
        setting     TEXT NOT NULL,
        previous    TEXT,
        candidate   TEXT,
        metric      TEXT NOT NULL DEFAULT 'tokens_per_second',
        baseline    REAL,
        measured    REAL,
        delta       REAL,
        status      TEXT NOT NULL DEFAULT 'trialled',
                    -- trialled | kept | reverted | refused | failed
        detail      TEXT
    )
    """,
    "CREATE INDEX idx_optimisations_setting ON optimisations (setting, id DESC)",
]

MIGRATION_0005 = [
    # Section 11: tools the system writes for itself. A tool is source code that runs
    # on this machine, so the table is built around the review: nothing is callable
    # until a human has read it, every version is kept, and every call is logged with
    # what went in and what came out.
    """
    CREATE TABLE tools (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL UNIQUE,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
        description   TEXT NOT NULL DEFAULT '',
        status        TEXT NOT NULL DEFAULT 'proposed',
                      -- proposed | approved | rejected | retired
        active_version TEXT,
        request_id    INTEGER,
        run_id        INTEGER
    )
    """,
    "CREATE INDEX idx_tools_status ON tools (status, name)",
    """
    CREATE TABLE tool_versions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_id      INTEGER NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
        version      TEXT NOT NULL,
        created_at   TEXT NOT NULL DEFAULT (datetime('now')),
        source       TEXT NOT NULL,
        parameters   TEXT NOT NULL DEFAULT '{}',
        capabilities TEXT NOT NULL DEFAULT '[]',
        examples     TEXT NOT NULL DEFAULT '[]',
        -- What the admission checks found: compiled, examples passed, stubbed version
        -- failed, declared capabilities within the allowance. Shown at review time.
        checks       TEXT NOT NULL DEFAULT '{}',
        status       TEXT NOT NULL DEFAULT 'proposed',
                     -- proposed | approved | rejected | superseded
        author       TEXT NOT NULL DEFAULT 'model',
        reviewed_at  TEXT,
        reviewed_by  TEXT,
        note         TEXT,
        run_id       INTEGER,
        fingerprint  TEXT NOT NULL UNIQUE,
        UNIQUE (tool_id, version)
    )
    """,
    "CREATE INDEX idx_tool_versions_status ON tool_versions (status, tool_id)",
    """
    CREATE TABLE tool_calls (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        tool_id     INTEGER REFERENCES tools(id) ON DELETE SET NULL,
        name        TEXT NOT NULL,
        version     TEXT,
        session_id  INTEGER,
        turn_id     INTEGER,
        arguments   TEXT,
        result      TEXT,
        ok          INTEGER NOT NULL DEFAULT 0,
        error       TEXT,
        duration_ms INTEGER,
        source      TEXT NOT NULL DEFAULT 'chat'   -- chat | review | cli | sleep
    )
    """,
    "CREATE INDEX idx_tool_calls_tool ON tool_calls (tool_id, id DESC)",
    """
    CREATE TABLE tool_requests (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        text       TEXT NOT NULL,
        source     TEXT NOT NULL DEFAULT 'user',
        status     TEXT NOT NULL DEFAULT 'open',
                   -- open | written | refused | withdrawn
        detail     TEXT,
        tool_id    INTEGER REFERENCES tools(id) ON DELETE SET NULL,
        run_id     INTEGER
    )
    """,
    "CREATE INDEX idx_tool_requests_open ON tool_requests (status, id)",
]

MIGRATION_0006 = [
    # Section 12: accounts, for when the TCP API is switched on. A caller on the Unix
    # socket is the owner — the filesystem already decided that — so these tables only
    # matter once the network can ask.
    """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL UNIQUE,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'user',   -- owner | user
        disabled      INTEGER NOT NULL DEFAULT 0,
        last_seen_at  TEXT,
        note          TEXT
    )
    """,
    # Tokens for the CLI. Only the hash is stored: a token is shown once, when it is
    # created, and cannot be recovered afterwards — only replaced.
    """
    CREATE TABLE api_tokens (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at   TEXT NOT NULL DEFAULT (datetime('now')),
        name         TEXT NOT NULL DEFAULT 'cli',
        token_hash   TEXT NOT NULL UNIQUE,
        last_used_at TEXT,
        expires_at   TEXT,
        revoked      INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX idx_api_tokens_user ON api_tokens (user_id, revoked)",
    # Browser sessions for the web interface. Same thing with an expiry.
    """
    CREATE TABLE web_sessions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at   TEXT NOT NULL DEFAULT (datetime('now')),
        token_hash   TEXT NOT NULL UNIQUE,
        last_seen_at TEXT,
        expires_at   TEXT,
        user_agent   TEXT
    )
    """,
    "CREATE INDEX idx_web_sessions_user ON web_sessions (user_id)",
    # A chat belongs to an account once there are accounts. NULL means "the owner, over
    # the socket", which is every chat that existed before this migration.
    "ALTER TABLE sessions ADD COLUMN user_id INTEGER REFERENCES users(id)",
    "ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0",
    "CREATE INDEX idx_sessions_user ON sessions (user_id, last_activity_at DESC)",
]

MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "initial_daemon_schema", MIGRATION_0001),
    (2, "weight_change_log", MIGRATION_0002),
    (3, "learning_store", MIGRATION_0003),
    (4, "expansion_and_optimisation", MIGRATION_0004),
    (5, "tools", MIGRATION_0005),
    (6, "accounts", MIGRATION_0006),
]
