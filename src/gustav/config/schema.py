"""Configuration schema.

One user-facing ``config.yaml``. Everything has a default, so an empty file is a
valid configuration. The ``tunable`` subtree is the *only* thing the daemon's own
self-optimisation is ever allowed to change (see section 10 of the spec).
"""

from __future__ import annotations

import re
from datetime import time as _time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_WINDOW_RE = re.compile(r"^(?P<h1>\d{1,2}):(?P<m1>\d{2})\s*-\s*(?P<h2>\d{1,2}):(?P<m2>\d{2})$")


class Strict(BaseModel):
    """Reject unknown keys: a typo in config.yaml should be an error, not a silent default."""

    model_config = ConfigDict(extra="forbid")


class TimeWindow(Strict):
    start: _time
    end: _time

    @classmethod
    def parse(cls, raw: str) -> "TimeWindow":
        m = _WINDOW_RE.match(raw.strip())
        if not m:
            raise ValueError(f"invalid time window {raw!r}, expected e.g. '00:00-06:00'")
        h1, m1, h2, m2 = (int(m.group(k)) for k in ("h1", "m1", "h2", "m2"))
        for h, mi in ((h1, m1), (h2, m2)):
            if not (0 <= h <= 23 and 0 <= mi <= 59):
                raise ValueError(f"invalid time window {raw!r}: {h:02d}:{mi:02d} is not a time")
        return cls(start=_time(h1, m1), end=_time(h2, m2))

    def contains(self, t: _time) -> bool:
        if self.start == self.end:  # degenerate: always open
            return True
        if self.start < self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end  # wraps midnight

    def __str__(self) -> str:
        return f"{self.start:%H:%M}-{self.end:%H:%M}"


class InstanceConfig(Strict):
    name: str = "gustav"
    data_dir: str = "var"
    #: Unix-socket control plane. Always on; this is how the CLI talks to the daemon.
    socket: str | None = None  # default: <data_dir>/gustav.sock
    #: Optional TCP listener, for remote clients / the dashboard.
    http_enabled: bool = False
    http_host: str = "127.0.0.1"
    http_port: int = 8713
    #: Token required for TCP requests (not for the socket, which is filesystem-protected).
    http_token: str | None = None


class LoggingConfig(Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str | None = None  # default: <data_dir>/logs/gustav.log
    json_lines: bool = False
    console: bool = True
    max_bytes: int = 32 * 1024 * 1024
    backups: int = 5


class ScheduleConfig(Strict):
    """When the daemon puts itself to sleep. Any combination of triggers."""

    enabled: bool = True
    #: Sleep windows in local time; empty list means "no window trigger".
    windows: list[str] = ["00:00-06:00"]
    #: Automatic sleep only ever happens inside a window unless a trigger says otherwise.
    idle_trigger_enabled: bool = True
    idle_minutes: int = 45
    pending_trigger_enabled: bool = True
    pending_threshold: int = 50
    #: Window and idle triggers need at least this much pending work; 0 = sleep anyway.
    min_pending_items: int = 1
    #: No automatic sleep until the user has been quiet this long (guards mid-conversation).
    min_idle_seconds: int = 120
    #: Don't start another sleep this soon after one finished (manual always allowed).
    cooldown_minutes: int = 60
    #: Hard cap on a single sleep run.
    max_duration_minutes: int = 300
    #: Pause training above this GPU temperature.
    max_gpu_temp_c: int = 83
    #: Stay awake while another process holds this much VRAM. A game, another model or
    #: a render job should not have to fight the daemon for the card.
    max_foreign_vram_mb: int = 2048
    #: Stay awake while the GPU is this busy with other software.
    max_foreign_gpu_percent: int = 35
    #: How often the scheduler re-evaluates its triggers.
    tick_seconds: float = 5.0

    @field_validator("windows")
    @classmethod
    def _check_windows(cls, v: list[str]) -> list[str]:
        for raw in v:
            TimeWindow.parse(raw)
        return v

    def parsed_windows(self) -> list[TimeWindow]:
        return [TimeWindow.parse(w) for w in self.windows]


class MaintenanceConfig(Strict):
    """Internal periodic timers (section 2: the daemon schedules its own upkeep)."""

    hardware_recheck_minutes: int = 60
    health_check_minutes: int = 5
    backup_retention_minutes: int = 360
    doc_cache_refresh_hours: int = 24
    report_minutes: int = 0  # 0 = only after a sleep run


class SupervisorConfig(Strict):
    """Child-process supervision: heavy work never runs in the daemon process."""

    heartbeat_timeout_seconds: float = 60.0
    #: Default wall-clock limit per step; a step may override it.
    step_timeout_seconds: float = 1800.0
    #: RLIMIT_AS for children, in MiB. 0 disables the limit.
    memory_limit_mb: int = 16384
    max_attempts: int = 3
    retry_backoff_seconds: float = 10.0
    retry_backoff_factor: float = 2.0
    #: Grace period between SIGTERM and SIGKILL.
    kill_grace_seconds: float = 10.0


class SleepConfig(Strict):
    """The sleep pipeline."""

    #: Steps the user can switch off entirely.
    disabled_steps: list[str] = []
    #: Steps that run the simulated placeholder handler instead of their real body.
    #: Empty now that every step is implemented; it stays because it is how an
    #: unfinished or deliberately disabled step keeps the night runnable, and because
    #: the tests of the pipeline machinery itself use it.
    simulated_steps: list[str] = []
    #: Seconds of simulated work per placeholder step.
    simulated_step_seconds: float = 0.5
    #: Replay share for anti-forgetting: this fraction of every training mix is drawn
    #: from examples that earlier runs already learned, so new data cannot quietly
    #: overwrite old skills.
    replay_fraction: float = 0.30
    #: How far back consolidation looks when nothing has been consolidated yet.
    lookback_hours: int = 72
    #: Per-night ceilings, so one busy day cannot produce a night that never ends.
    max_facts_per_run: int = 200
    max_examples_per_run: int = 2000
    #: Share of new examples held back for the gate. These are never trained on.
    heldout_fraction: float = 0.15
    #: Topics `reflect` may pick per night, and the pass rate below which a topic is weak.
    max_topics_per_run: int = 5
    weak_score_threshold: float = 0.6
    #: Use the model for consolidation and study. With it off, only the deterministic
    #: extractors run — much less data, but no GPU and no model errors in the dataset.
    use_model: bool = True
    #: Tokens the model may spend per consolidation/study/practice answer.
    max_new_tokens: int = 384


class WakeConfig(Strict):
    """The chat side."""

    #: Release the GPU after this long without a request, so the VRAM is free for
    #: whatever else you are doing. The next message reloads the model.
    #: 0 disables it and keeps the model resident.
    idle_unload_minutes: int = 15
    session_idle_minutes: int = 120
    max_history_turns: int = 40
    #: When the conversation no longer fits the window, fold the turns that fall off
    #: into one summary message instead of dropping them. It costs one short generation
    #: per conversation (cached until more turns fall off), and it is the difference
    #: between "we were discussing the deploy script" and an answer that has forgotten
    #: the question.
    summarise_history: bool = True
    summary_tokens: int = 160
    #: Store a redacted copy when a secret is detected. The live request still sees the
    #: original text; only what is persisted changes. Turning this off keeps the raw
    #: text on disk — it is still never trained on.
    redact_secrets: bool = True


class TunableConfig(Strict):
    """The ONLY values self-optimisation may change. Never source code."""

    kv_cache_dtype: Literal["auto", "fp16", "int8", "fp8"] = "auto"
    batch_size: int = 1
    speculative_decoding: bool = False
    router_threshold: float = 0.45
    #: Reuse the KV cache for the shared prefix of consecutive prompts. A large win on
    #: multi-turn chat (the prompt is re-sent every turn), at the cost of bit-exact
    #: reproducibility: a reused cache computes the prefill differently, so a greedy
    #: answer can differ by a token from the same question asked cold.
    prefix_cache: bool = True
    prefix_cache_mb: int = 512
    warm_cache_mb: int = 8192


class ToolsConfig(Strict):
    """Section 11: tools the system writes for itself.

    A tool is *source code that runs on this machine*, which makes it the most
    dangerous thing in the project — so the defaults are the cautious ones and the
    review is not optional. The sandbox is described in `gustav/tools/policy.py`,
    including what it does and does not guarantee.
    """

    #: May the model call approved tools at all?
    enabled: bool = True
    #: May the night write new ones? Turning this off leaves the tools you approved
    #: working and stops new ones appearing.
    allow_authoring: bool = True
    #: Nothing is callable until a human has read the source. Turning this off is a
    #: deliberate act and the daemon says so every time it runs a tool.
    require_approval: bool = True
    max_tools: int = 20
    #: Tool calls one answer may make, so a loop cannot turn into a night.
    max_calls_per_turn: int = 4
    #: Per call.
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 512
    max_output_bytes: int = 64 * 1024
    #: Tools a single night may write.
    max_new_per_run: int = 2
    max_source_bytes: int = 20_000
    #: What a tool may ever ask for. A tool that declares anything else is refused
    #: before it reaches review.
    #:
    #: ``pure``            compute only: no files, no network, no subprocesses
    #: ``write_workspace`` read and write inside its own throwaway directory
    #: ``read_paths``      read files under `tools.readable_paths`
    #: ``network``         HTTP to `tools.allowed_hosts`, and nothing else
    allow_capabilities: list[str] = ["pure", "write_workspace"]
    #: Modules a tool may import. Enforced twice: statically before review, and at
    #: runtime by an audit hook that cannot be removed once installed.
    allowed_modules: list[str] = [
        "base64", "bisect", "calendar", "cmath", "collections", "colorsys", "copy",
        "csv", "dataclasses", "datetime", "decimal", "difflib", "enum", "fnmatch",
        "fractions", "functools", "hashlib", "heapq", "hmac", "html", "io",
        "ipaddress", "itertools", "json", "math", "operator", "pathlib", "pprint",
        "random", "re", "secrets", "statistics", "string", "textwrap", "time",
        "timeit", "types", "typing", "unicodedata", "urllib.parse", "uuid",
        "zoneinfo",
    ]
    #: Directories a `read_paths` tool may read. Empty means the capability is useless,
    #: which is the right default: your project directory holds your configuration.
    readable_paths: list[str] = []
    #: Hosts a `network` tool may reach. Empty means no tool can use the network.
    allowed_hosts: list[str] = []


class ExpansionConfig(Strict):
    """Section 8: growing its own capacity, within a budget the hardware sets.

    Every knob here is a brake. The triggers are measurements the system already takes
    — requests no module covered, a module that stopped improving, one nobody uses —
    and the defaults are chosen so that the cheap, reversible changes happen by
    themselves and the expensive, destructive ones wait for you.
    """

    enabled: bool = True
    #: Structural changes per night. One, so that if something breaks tomorrow the
    #: cause is a single named change.
    max_changes_per_run: int = 1
    #: A new module needs this many requests nothing covered, sharing enough vocabulary
    #: to be one subject rather than a pile of unrelated questions.
    min_uncovered_requests: int = 8
    min_cluster_share: float = 0.4
    #: A module is "saturated" when this many consecutive gate results moved the
    #: held-out score by less than `saturation_delta`. More rank is then worth trying.
    saturation_runs: int = 3
    saturation_delta: float = 0.01
    rank_step: int = 8
    #: A module with an adapter, no demand and no activations for this long is a
    #: candidate for archiving (never deletion).
    unused_days: int = 30
    allow_new_modules: bool = True
    allow_rank_increase: bool = True
    #: Off by default: archiving a module changes what the system can do.
    allow_pruning: bool = False
    #: Merging an adapter into the base rewrites the base checkpoint. It is the one
    #: operation that can lose the weights everything else is built on, so it is off,
    #: and approval-gated even when on.
    allow_base_merge: bool = False
    #: Kinds that may never be applied without you saying so.
    require_approval: list[str] = ["merge_to_base", "prune_module", "new_module"]


class OptimisationConfig(Strict):
    """Section 10: tuning itself, by measurement, inside `tunable` and nowhere else."""

    enabled: bool = True
    #: Settings it may try. Anything not listed is out of reach even though it lives in
    #: `tunable`, and anything pinned in config.yaml wins over whatever it learns.
    settings: list[str] = [
        "prefix_cache",
        "kv_cache_dtype",
        "warm_cache_mb",
        "speculative_decoding",
    ]
    #: Trials per night, and how long one measurement may take.
    max_trials_per_run: int = 2
    bench_tokens: int = 96
    bench_repeats: int = 2
    #: Keep a change only if it is this much better, in relative terms. Noise is the
    #: enemy here: a 1% "win" measured once is not a win.
    min_relative_gain: float = 0.05
    #: Never trade this much answer quality for speed; measured on the gate's suites.
    max_quality_loss: float = 0.02


class HardwareConfig(Strict):
    """How the planner should size the system for this machine."""

    #: "auto" runs the planner; any other name loads profiles/<name>.yaml.
    profile: str = "auto"
    #: 0.0 = as fast as possible, 1.0 = as capable as possible.
    balance: float = Field(default=0.5, ge=0.0, le=1.0)
    #: The chat speed we refuse to go below.
    min_tokens_per_second: float = 15.0
    #: Left free on every GPU for other software.
    vram_reserve_fraction: float = Field(default=0.15, ge=0.0, le=0.9)
    ram_reserve_mb: int = 8192
    disk_reserve_mb: int = 10240
    allow_training: bool = True
    #: When true, a model is only chosen if QLoRA training also fits.
    require_training: bool = True
    #: Empty means "decide from the detected GPUs": with two or more cards, one serves
    #: and the rest train, so chat stays up during sleep.
    inference_gpus: list[int] = []
    training_gpus: list[int] = []
    max_seq_len: int | None = None
    train_seq_len: int | None = None
    kv_cache_dtype: Literal["auto", "fp16", "bf16", "int8", "fp8"] = "auto"
    speculative_decoding: Literal["auto", "on", "off"] = "auto"
    #: Pin the base model, or restrict/exclude ladder entries.
    prefer_model: str | None = None
    allow_models: list[str] = []
    deny_models: list[str] = []
    #: Alternative ladder file.
    ladder: str | None = None
    #: Re-plan when the hardware fingerprint changes (never silently: you are told).
    replan_on_hardware_change: bool = True
    #: A plan written here (or in a profile) wins over anything the planner computes.
    plan: dict[str, Any] = {}


class BackupConfig(Strict):
    """Section 9. Nothing that changes weights runs without a verified snapshot."""

    #: Turning this off removes the safety net. It exists for a machine that backs the
    #: whole tree up by other means, and the daemon says so every time it is used.
    enabled: bool = True
    #: Where snapshots go. Ideally another disk, or a remote path. Default: <root>/backups.
    location: str | None = None
    #: Retention: the last N, plus one a day for M days and one a week for K weeks.
    keep_last: int = 5
    daily_days: int = 7
    weekly_weeks: int = 4
    #: Refuse to take a snapshot that would leave less than this free.
    min_free_mb: int = 5120
    #: Warn (rather than delete) when the location gets this low.
    warn_free_mb: int = 20480
    #: Include the base checkpoint. Only needed for a merge or a base change, which ask
    #: for it explicitly; it is several gigabytes.
    include_base_model: bool = False
    verify_after_write: bool = True


class FetcherConfig(Strict):
    """The one component that touches the network, and the allowlist that limits it."""

    enabled: bool = True
    #: Hosts documentation may be fetched from. Nothing else is ever requested, and a
    #: redirect that leaves the list is dropped rather than followed.
    allowed_hosts: list[str] = [
        "docs.python.org",
        "doc.rust-lang.org",
        "docs.rs",
        "developer.mozilla.org",
        "pkg.go.dev",
        "peps.python.org",
    ]
    #: Extra local directories to study from. Useful offline, and the only source used
    #: when `enabled` is false.
    local_sources: list[str] = []
    user_agent: str = "gustav-docs/0.1 (local documentation fetcher)"
    timeout_seconds: float = 20.0
    max_bytes_per_document: int = 2 * 1024 * 1024
    max_documents_per_run: int = 12
    #: Don't re-fetch something already in the cache this recently.
    cache_ttl_hours: int = 168
    #: Be a polite client: this many seconds between requests to the same host.
    delay_seconds: float = 1.0


class SandboxConfig(Strict):
    """Where practice answers are judged. Nothing here trusts the model's own opinion."""

    #: Interpreter for python tasks. Default: the one running the daemon.
    python: str | None = None
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 1024
    #: Hard cap on how much a verifier may print back.
    max_output_bytes: int = 64 * 1024
    #: Candidate code runs with no network namespace access we can grant it anyway;
    #: this clears proxy variables and sets a throwaway HOME and cwd per attempt.
    isolate_env: bool = True


class TrainingConfig(Strict):
    """LoRA training. Everything here is a ceiling, not a target."""

    enabled: bool = True
    #: LoRA shape. `rank: null` takes the rank from the hardware plan.
    rank: int | None = None
    alpha_ratio: float = 2.0
    dropout: float = 0.05
    learning_rate: float = 1e-4
    #: Effective batch = batch_size * grad_accum; the plan sizes memory for batch_size.
    batch_size: int = 1
    grad_accum: int = 8
    epochs: float = 1.0
    #: Hard caps so a night cannot run away: whichever is hit first stops training.
    max_steps: int = 200
    max_seconds: float = 1800.0
    #: Don't bother training a module with less than this much new material.
    min_examples: int = 16
    #: Train only on examples something other than the model agreed with: a correction,
    #: a thumbs-up, a passing test. The strict setting — it throws away most of what a
    #: night produces (cited study answers, self-consistent rewrites) in exchange for a
    #: dataset where every row has an external authority behind it.
    verified_only: bool = False
    #: Modules to train. Empty = every module with enough examples.
    modules: list[str] = []
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    #: How preference pairs (a `/good`, a `/bad` the night replaced, a rewrite the
    #: samples agreed on) are trained.
    #:
    #: ``sft``  fit the preferred answer and ignore the rejected one — simple, and it
    #:          cannot push probability mass anywhere in particular.
    #: ``dpo``  direct preference optimisation against the base model as the reference,
    #:          which is what the stored pairs are shaped for. It costs four forward
    #:          passes per pair instead of one, so it is measured by the gate like
    #:          everything else.
    preference_objective: Literal["sft", "dpo"] = "sft"
    dpo_beta: float = 0.1
    #: Save a checkpoint this often, so a preempted night resumes rather than restarts.
    checkpoint_every_steps: int = 25


class GateConfig(Strict):
    """The regression gate. Nothing is promoted unless this passes."""

    enabled: bool = True
    #: A candidate must beat the baseline on the held-out suite by at least this much.
    min_improvement: float = 0.0
    #: And may not lose more than this on the regression suite — the anti-forgetting
    #: check. 0.02 is one item in fifty, i.e. noise, not a skill.
    max_regression: float = 0.02
    #: Below this many items a score is not a measurement; the gate skips instead of
    #: promoting on three questions.
    min_items: int = 8
    max_items: int = 120
    max_new_tokens: int = 256
    #: How a free-text answer is compared with its reference.
    #:
    #: ``embedding`` uses mean-pooled hidden states from the model already in VRAM, so
    #: a right answer in different words still counts. ``overlap`` compares content
    #: words, which is cruder and has no dependencies. "embedding" falls back to
    #: "overlap" when the backend cannot embed, and the report says which was used.
    scorer: Literal["embedding", "overlap"] = "embedding"
    embedding_match: float = 0.72
    #: Promote only one module per night by default: if something breaks, the cause is
    #: unambiguous. Raise it once the gate has earned your trust.
    max_promotions_per_run: int = 1


class Config(Strict):
    instance: InstanceConfig = InstanceConfig()
    logging: LoggingConfig = LoggingConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    maintenance: MaintenanceConfig = MaintenanceConfig()
    supervisor: SupervisorConfig = SupervisorConfig()
    sleep: SleepConfig = SleepConfig()
    wake: WakeConfig = WakeConfig()
    tunable: TunableConfig = TunableConfig()
    hardware: HardwareConfig = HardwareConfig()
    #: Sections owned by later stages. Declared loosely now so a stage-3+ config
    #: file still loads under this build.
    backup: BackupConfig = BackupConfig()
    inference: dict[str, Any] = {}
    tools: ToolsConfig = ToolsConfig()
    expansion: ExpansionConfig = ExpansionConfig()
    optimisation: OptimisationConfig = OptimisationConfig()
    fetcher: FetcherConfig = FetcherConfig()
    sandbox: SandboxConfig = SandboxConfig()
    training: TrainingConfig = TrainingConfig()
    gate: GateConfig = GateConfig()
