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
    """The sleep pipeline. Stage 1 ships a journalled skeleton; real stages land later."""

    #: Steps the user can switch off entirely.
    disabled_steps: list[str] = []
    #: Stage-1 placeholder pipeline: seconds of simulated work per step.
    simulated_step_seconds: float = 0.5
    #: Replay share for anti-forgetting (used from stage 5).
    replay_fraction: float = 0.30


class WakeConfig(Strict):
    session_idle_minutes: int = 120
    max_history_turns: int = 40


class TunableConfig(Strict):
    """The ONLY values self-optimisation may change. Never source code."""

    kv_cache_dtype: Literal["auto", "fp16", "int8", "fp8"] = "auto"
    batch_size: int = 1
    speculative_decoding: bool = False
    router_threshold: float = 0.45
    prefix_cache_mb: int = 512
    warm_cache_mb: int = 8192


class Config(Strict):
    instance: InstanceConfig = InstanceConfig()
    logging: LoggingConfig = LoggingConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    maintenance: MaintenanceConfig = MaintenanceConfig()
    supervisor: SupervisorConfig = SupervisorConfig()
    sleep: SleepConfig = SleepConfig()
    wake: WakeConfig = WakeConfig()
    tunable: TunableConfig = TunableConfig()
    #: Sections owned by later stages. Declared loosely now so a stage-2+ config
    #: file still loads under a stage-1 build.
    hardware: dict[str, Any] = {}
    inference: dict[str, Any] = {}
    backup: dict[str, Any] = {}
    expansion: dict[str, Any] = {}
    fetcher: dict[str, Any] = {}
    sandbox: dict[str, Any] = {}
