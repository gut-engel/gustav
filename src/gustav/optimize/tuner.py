"""Self-optimisation: tuning itself, by measurement, inside `tunable` and nowhere else.

Section 10 draws a hard line: the daemon may change the values in the ``tunable``
section of its configuration, and nothing else — never its own source, never the
hardware plan, never a threshold that decides what gets learned. This module is the
only thing that writes those values, and it writes them to a file of its own
(``var/tunable.yaml``) that sits *underneath* your ``config.yaml``, so anything you
pin stays pinned.

The loop is deliberately boring, because the failure mode of self-optimisation is
believing noise:

1. measure the current setting (a short, fixed benchmark);
2. change one value;
3. measure again — and, for anything that can change what the model *says*, measure
   answer quality too;
4. keep it only if it is better by ``min_relative_gain`` and no worse in quality by
   more than ``max_quality_loss``; otherwise put it back and record why.

A trial that was reverted is remembered, so the same losing value is not re-tried every
night for the rest of time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gustav.config.load import read_learned_tunable, write_learned_tunable
from gustav.config.schema import OptimisationConfig
from gustav.core.events import Events
from gustav.paths import Paths
from gustav.store.db import Database

#: What each setting may be tried at.
#:
#: Only settings the serving path actually reads are here. `batch_size` is not: requests
#: are served one at a time, so "tuning" it would measure noise and keep the winner by
#: luck. It comes back when there is continuous batching to tune.
CANDIDATES: dict[str, list[Any]] = {
    "prefix_cache": [True, False],
    "kv_cache_dtype": ["auto", "fp16", "int8"],
    "warm_cache_mb": [2048, 4096, 8192],
    "speculative_decoding": [True, False],
}

#: Settings that can change the tokens that come out, not just how fast they come out.
#: These are measured for quality as well as speed, and a quality loss vetoes the gain.
#:
#: Speculative decoding is deliberately *not* here: for greedy decoding it accepts only
#: tokens the target model would have produced anyway, so the output is identical by
#: construction and a quality measurement would just be measuring noise. The test that
#: holds that claim up is the one asserting the two paths produce the same text.
QUALITY_SENSITIVE = frozenset({"kv_cache_dtype", "prefix_cache"})


@dataclass
class Measurement:
    """One observation of the system under a given setting."""

    tokens_per_second: float
    quality: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trial:
    setting: str
    previous: Any
    candidate: Any
    baseline: Measurement | None = None
    measured: Measurement | None = None
    status: str = "trialled"
    detail: str = ""

    @property
    def gain(self) -> float | None:
        """Relative speed change, or None if either side was not measured."""
        if self.baseline is None or self.measured is None:
            return None
        if not self.baseline.tokens_per_second:
            return None
        return (
            self.measured.tokens_per_second - self.baseline.tokens_per_second
        ) / self.baseline.tokens_per_second

    @property
    def quality_loss(self) -> float | None:
        if (
            self.baseline is None
            or self.measured is None
            or self.baseline.quality is None
            or self.measured.quality is None
        ):
            return None
        return self.baseline.quality - self.measured.quality

    def describe(self) -> str:
        gain = f"{self.gain:+.1%}" if self.gain is not None else "unmeasured"
        return (
            f"{self.setting}: {self.previous!r} -> {self.candidate!r} "
            f"({gain}) — {self.status}"
            + (f": {self.detail}" if self.detail else "")
        )


@dataclass
class TuneResult:
    baseline: Measurement | None = None
    trials: list[Trial] = field(default_factory=list)
    kept: dict[str, Any] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.trials:
            return "nothing was worth trying"
        kept = ", ".join(f"{k}={v!r}" for k, v in self.kept.items()) or "nothing"
        return f"{len(self.trials)} trial(s), kept: {kept}"


class Tuner:
    def __init__(
        self,
        *,
        config: OptimisationConfig,
        tunable: Any,
        paths: Paths,
        db: Database,
        events: Events,
        pinned: set[str] | None = None,
        logger: Any = None,
    ):
        from gustav.core.logs import get_logger

        self.config = config
        self.tunable = tunable
        self.paths = paths
        self.db = db
        self.events = events
        self.pinned = pinned or set()
        self.log = logger or get_logger("optimise")

    # --- choosing what to try -------------------------------------------
    def current(self, setting: str) -> Any:
        return getattr(self.tunable, setting, None)

    def tried(self, setting: str, value: Any) -> str | None:
        """The verdict the last time this exact value was measured, if ever."""
        row = self.db.query_one(
            "SELECT status FROM optimisations WHERE setting = ? AND candidate = ?"
            " ORDER BY id DESC LIMIT 1",
            (setting, repr(value)),
        )
        return row["status"] if row else None

    def candidates(self, *, supported: dict[str, list[Any]] | None = None) -> list[tuple[str, Any]]:
        """(setting, value) pairs worth a measurement tonight.

        Skipped: settings you pinned, settings not in `optimisation.settings`, the value
        already in use, and any value that has already been measured and lost.
        """
        out: list[tuple[str, Any]] = []
        table = supported or CANDIDATES
        for setting in self.config.settings:
            if setting in self.pinned:
                continue
            if setting not in table:
                continue
            current = self.current(setting)
            for value in table[setting]:
                if value == current:
                    continue
                verdict = self.tried(setting, value)
                if verdict in ("reverted", "refused", "failed"):
                    continue
                out.append((setting, value))
        return out

    def why_skipped(self) -> list[str]:
        reasons = []
        for setting in self.config.settings:
            if setting in self.pinned:
                reasons.append(f"{setting}: pinned in config.yaml")
        return reasons

    # --- the loop -------------------------------------------------------
    def run(
        self,
        measure: Callable[[dict[str, Any]], Measurement],
        *,
        run_id: int | None = None,
        supported: dict[str, list[Any]] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> TuneResult:
        """Measure the current settings, then try a few changes against them."""
        result = TuneResult(skipped=self.why_skipped())
        if not self.config.enabled:
            return result
        pairs = self.candidates(supported=supported)[: self.config.max_trials_per_run]
        if not pairs:
            return result

        def say(message: str) -> None:
            if progress is not None:
                progress(message)

        say("measuring the current settings")
        try:
            result.baseline = measure({})
        except Exception as exc:  # noqa: BLE001 - a failed baseline means no trials
            self.log.warning("baseline measurement failed: %s", exc)
            result.skipped.append(f"baseline measurement failed: {exc}")
            return result

        learned = read_learned_tunable(self.paths.data_dir)
        for setting, value in pairs:
            trial = Trial(
                setting=setting,
                previous=self.current(setting),
                candidate=value,
                baseline=result.baseline,
            )
            say(f"trying {setting}={value!r}")
            try:
                trial.measured = measure({setting: value})
            except Exception as exc:  # noqa: BLE001
                trial.status = "failed"
                trial.detail = f"{type(exc).__name__}: {exc}"
                self.record(trial, run_id=run_id)
                result.trials.append(trial)
                continue

            self._judge(trial)
            if trial.status == "kept":
                learned[setting] = value
                write_learned_tunable(self.paths.data_dir, learned)
                setattr(self.tunable, setting, value)
                result.kept[setting] = value
                # The baseline moves with it: later trials tonight are measured
                # against the configuration now in force, not the one we started from.
                result.baseline = trial.measured
                self.events.emit(
                    "optimise.kept",
                    trial.describe(),
                    setting=setting,
                    value=repr(value),
                    gain=round(trial.gain or 0.0, 4),
                )
            else:
                self.events.emit(
                    "optimise.reverted",
                    trial.describe(),
                    setting=setting,
                    value=repr(value),
                )
            self.record(trial, run_id=run_id)
            result.trials.append(trial)
        return result

    def _judge(self, trial: Trial) -> None:
        gain = trial.gain
        loss = trial.quality_loss
        if gain is None:
            trial.status = "refused"
            trial.detail = "the measurement produced no speed to compare"
            return
        if loss is not None and loss > self.config.max_quality_loss:
            trial.status = "reverted"
            trial.detail = (
                f"{gain:+.1%} faster but {loss:.1%} worse on the held-out suite, over the "
                f"{self.config.max_quality_loss:.1%} allowed"
            )
            return
        if gain < self.config.min_relative_gain:
            trial.status = "reverted"
            trial.detail = (
                f"{gain:+.1%} is below the {self.config.min_relative_gain:.1%} that "
                f"counts as a win"
            )
            return
        trial.status = "kept"
        trial.detail = f"{gain:+.1%} faster" + (
            f", quality {(-loss):+.1%}" if loss is not None else ""
        )

    # --- the record -----------------------------------------------------
    def record(self, trial: Trial, *, run_id: int | None = None) -> int:
        return self.db.insert(
            "optimisations",
            {
                "run_id": run_id,
                "setting": trial.setting,
                "previous": repr(trial.previous),
                "candidate": repr(trial.candidate),
                "metric": "tokens_per_second",
                "baseline": trial.baseline.tokens_per_second if trial.baseline else None,
                "measured": trial.measured.tokens_per_second if trial.measured else None,
                "delta": trial.gain,
                "status": trial.status,
                "detail": trial.detail[:1000],
            },
        )

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query(
                "SELECT * FROM optimisations ORDER BY id DESC LIMIT ?", (limit,)
            )
        ]

    # --- taking it back -------------------------------------------------
    def revert_last(self) -> dict[str, Any]:
        """Undo the most recently kept change, and remember not to try it again."""
        row = self.db.query_one(
            "SELECT * FROM optimisations WHERE status = 'kept' ORDER BY id DESC LIMIT 1"
        )
        if row is None:
            raise KeyError("self-optimisation has not kept any change to undo")
        setting = row["setting"]
        learned = read_learned_tunable(self.paths.data_dir)
        removed = learned.pop(setting, None)
        write_learned_tunable(self.paths.data_dir, learned)
        # Put the live value back too, not just the file: the daemon re-reads its
        # configuration after a revert, but anything holding this Tuner would otherwise
        # keep offering the old value as the one "in use".
        import ast

        try:
            setattr(self.tunable, setting, ast.literal_eval(row["previous"] or ""))
        except (ValueError, SyntaxError):
            self.log.warning(
                "could not restore %s in memory from %r; it is correct on disk",
                setting, row["previous"],
            )
        self.db.execute(
            "UPDATE optimisations SET status = 'reverted', detail = ? WHERE id = ?",
            ("reverted by hand", int(row["id"])),
        )
        self.events.emit(
            "optimise.reverted",
            f"{setting} was put back (it had been set to {row['candidate']})",
            level="warning",
            setting=setting,
        )
        return {
            "setting": setting,
            "was": row["candidate"],
            "restored": row["previous"],
            "removed": removed,
            "learned": learned,
        }
