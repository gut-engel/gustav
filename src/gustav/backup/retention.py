"""Retention: keep the last N, one a day for M days, one a week for K weeks.

Two things are never deleted, whatever the policy says: the newest verified snapshot,
and any snapshot a pending approval refers to. And when the location is nearly full the
answer is a warning, not a quiet deletion — losing backups to free space is exactly the
failure this is supposed to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from gustav.backup.snapshot import Snapshot


@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int = 5
    daily_days: int = 7
    weekly_weeks: int = 4
    #: Incomplete snapshots are rubbish from an interrupted backup; clear them sooner.
    incomplete_after_hours: int = 6

    def describe(self) -> str:
        return (
            f"keep the last {self.keep_last}, one daily for {self.daily_days} days, "
            f"one weekly for {self.weekly_weeks} weeks"
        )


@dataclass
class RetentionDecision:
    keep: list[Snapshot] = field(default_factory=list)
    delete: list[Snapshot] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    def describe(self) -> list[str]:
        lines = [f"keeping {len(self.keep)}, deleting {len(self.delete)}"]
        for snapshot in self.keep:
            lines.append(f"  keep   {snapshot.id:<34} {self.reasons.get(snapshot.id, '')}")
        for snapshot in self.delete:
            lines.append(f"  delete {snapshot.id:<34} {self.reasons.get(snapshot.id, '')}")
        return lines


def _parse(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def plan_retention(
    snapshots: Iterable[Snapshot],
    policy: RetentionPolicy,
    *,
    protected: set[str] | None = None,
    now: datetime | None = None,
) -> RetentionDecision:
    now = now or datetime.now()
    protected = protected or set()
    ordered = sorted(snapshots, key=lambda s: (s.created_at, s.id), reverse=True)
    decision = RetentionDecision()

    keep_ids: dict[str, str] = {}
    seen_days: set[str] = set()
    seen_weeks: set[str] = set()

    newest_complete = next((s for s in ordered if s.complete), None)
    if newest_complete is not None:
        keep_ids[newest_complete.id] = "the newest complete snapshot is never deleted"

    for index, snapshot in enumerate(ordered):
        if snapshot.id in protected:
            keep_ids.setdefault(snapshot.id, "referenced by a pending approval")
            continue
        if not snapshot.complete:
            created = _parse(snapshot.created_at)
            age_hours = (now - created).total_seconds() / 3600 if created else 999
            if age_hours > policy.incomplete_after_hours:
                continue  # rubbish from an interrupted backup
            keep_ids.setdefault(snapshot.id, "incomplete but recent")
            continue
        if index < policy.keep_last:
            keep_ids.setdefault(snapshot.id, f"one of the last {policy.keep_last}")
            continue

        created = _parse(snapshot.created_at)
        if created is None:
            keep_ids.setdefault(snapshot.id, "undated, kept to be safe")
            continue
        age = now - created
        day_key = created.strftime("%Y-%m-%d")
        week_key = created.strftime("%G-W%V")
        if age <= timedelta(days=policy.daily_days) and day_key not in seen_days:
            seen_days.add(day_key)
            keep_ids.setdefault(snapshot.id, f"the daily snapshot for {day_key}")
            continue
        if age <= timedelta(weeks=policy.weekly_weeks) and week_key not in seen_weeks:
            seen_weeks.add(week_key)
            keep_ids.setdefault(snapshot.id, f"the weekly snapshot for {week_key}")
            continue

    for snapshot in ordered:
        if snapshot.id in keep_ids:
            decision.keep.append(snapshot)
            decision.reasons[snapshot.id] = keep_ids[snapshot.id]
        else:
            decision.delete.append(snapshot)
            decision.reasons[snapshot.id] = (
                "incomplete and stale" if not snapshot.complete else "outside the policy"
            )
    return decision
