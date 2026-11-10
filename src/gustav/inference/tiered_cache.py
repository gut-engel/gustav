"""Tiered adapter storage: VRAM (hot) -> pinned RAM (warm) -> disk (cold).

This is the *policy*, not the plumbing: it tracks what lives where, enforces the budget
from the hardware plan, and decides what to evict to make room. The backend performs the
moves it asks for. Keeping the decision separate means the LRU and the budget arithmetic
are testable without a GPU, and the budget is never exceeded by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Tier(StrEnum):
    VRAM = "vram"
    RAM = "ram"
    DISK = "disk"

    @property
    def rank(self) -> int:
        return {Tier.VRAM: 0, Tier.RAM: 1, Tier.DISK: 2}[self]


@dataclass
class Entry:
    name: str
    size_mb: float
    tier: Tier = Tier.DISK
    last_used: float = 0.0
    hits: int = 0
    promotions: int = 0
    evictions: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Move:
    name: str
    frm: Tier
    to: Tier

    def describe(self) -> str:
        return f"{self.name}: {self.frm} -> {self.to}"


@dataclass
class Admission:
    """What has to happen for a module to become resident."""

    name: str
    target: Tier
    moves: list[Move] = field(default_factory=list)
    admitted: bool = True
    reason: str = ""

    @property
    def evictions(self) -> list[Move]:
        return [m for m in self.moves if m.name != self.name]


class CacheFull(RuntimeError):
    pass


class TieredCache:
    def __init__(
        self,
        *,
        vram_budget_mb: float,
        ram_budget_mb: float = 0.0,
        clock=None,
    ):
        from gustav.core.clock import RealClock

        self.vram_budget_mb = float(vram_budget_mb)
        self.ram_budget_mb = float(ram_budget_mb)
        self.clock = clock or RealClock()
        self.entries: dict[str, Entry] = {}
        self.hits = 0
        self.misses = 0

    # --- bookkeeping ----------------------------------------------------
    def register(self, name: str, size_mb: float, *, tier: Tier = Tier.DISK, **meta: Any) -> Entry:
        entry = self.entries.get(name)
        if entry is None:
            entry = Entry(name=name, size_mb=float(size_mb), tier=tier, meta=meta)
            self.entries[name] = entry
        else:
            entry.size_mb = float(size_mb)
            entry.meta.update(meta)
        return entry

    def forget(self, name: str) -> None:
        self.entries.pop(name, None)

    def touch(self, name: str) -> None:
        entry = self.entries.get(name)
        if entry is None:
            return
        entry.last_used = self.clock.monotonic()
        entry.hits += 1

    def tier_of(self, name: str) -> Tier:
        entry = self.entries.get(name)
        return entry.tier if entry else Tier.DISK

    def usage_mb(self, tier: Tier) -> float:
        return round(sum(e.size_mb for e in self.entries.values() if e.tier is tier), 2)

    def resident(self, tier: Tier = Tier.VRAM) -> list[str]:
        return [e.name for e in self.entries.values() if e.tier is tier]

    def budget_for(self, tier: Tier) -> float:
        return self.vram_budget_mb if tier is Tier.VRAM else self.ram_budget_mb

    def free_mb(self, tier: Tier) -> float:
        return round(self.budget_for(tier) - self.usage_mb(tier), 2)

    # --- admission ------------------------------------------------------
    def plan_admission(
        self, name: str, *, target: Tier = Tier.VRAM, keep: tuple[str, ...] = ()
    ) -> Admission:
        """What to evict so that ``name`` fits in ``target``. Nothing is moved yet."""
        entry = self.entries.get(name)
        if entry is None:
            return Admission(
                name=name, target=target, admitted=False,
                reason=f"{name} is not registered in the cache",
            )
        if entry.tier.rank <= target.rank:
            self.hits += 1
            return Admission(name=name, target=entry.tier, reason="already resident")

        self.misses += 1
        if entry.size_mb > self.budget_for(target):
            return Admission(
                name=name, target=target, admitted=False,
                reason=(
                    f"{name} needs {entry.size_mb:.0f} MiB but the whole {target} budget is "
                    f"{self.budget_for(target):.0f} MiB"
                ),
            )

        moves: list[Move] = []
        freed = self.free_mb(target)
        protected = {*keep, name}
        if freed < entry.size_mb:
            # Least recently used first: that is the whole point of the tier.
            candidates = sorted(
                (e for e in self.entries.values() if e.tier is target and e.name not in protected),
                key=lambda e: (e.last_used, -e.size_mb),
            )
            for candidate in candidates:
                demoted = Tier.RAM if target is Tier.VRAM else Tier.DISK
                moves.append(Move(candidate.name, candidate.tier, demoted))
                freed += candidate.size_mb
                if freed >= entry.size_mb:
                    break
        if freed < entry.size_mb:
            return Admission(
                name=name, target=target, admitted=False,
                reason=(
                    f"cannot free {entry.size_mb:.0f} MiB in {target}: "
                    f"{freed:.0f} MiB available after evicting everything evictable"
                ),
            )
        moves.append(Move(name, entry.tier, target))
        return Admission(name=name, target=target, moves=moves, reason="promoted")

    def apply(self, admission: Admission) -> Admission:
        """Record the moves as done. The backend has already performed them."""
        if not admission.admitted:
            return admission
        for move in admission.moves:
            entry = self.entries.get(move.name)
            if entry is None:
                continue
            entry.tier = move.to
            if move.to.rank < move.frm.rank:
                entry.promotions += 1
                entry.last_used = self.clock.monotonic()
            else:
                entry.evictions += 1
        return admission

    def admit(self, name: str, *, target: Tier = Tier.VRAM, keep: tuple[str, ...] = ()) -> Admission:
        """Plan and apply in one step (when the caller has nothing to do in between)."""
        return self.apply(self.plan_admission(name, target=target, keep=keep))

    def require(self, names: tuple[str, ...], *, target: Tier = Tier.VRAM) -> list[Admission]:
        """Make a whole set resident, keeping the others in the set protected."""
        out: list[Admission] = []
        for name in names:
            out.append(self.admit(name, target=target, keep=names))
        return out

    # --- reporting ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "vram": {
                "used_mb": self.usage_mb(Tier.VRAM),
                "budget_mb": round(self.vram_budget_mb, 1),
                "modules": self.resident(Tier.VRAM),
            },
            "ram": {
                "used_mb": self.usage_mb(Tier.RAM),
                "budget_mb": round(self.ram_budget_mb, 1),
                "modules": self.resident(Tier.RAM),
            },
            "disk": {"modules": self.resident(Tier.DISK)},
            "hit_rate": round(self.hits / total, 3) if total else None,
            "hits": self.hits,
            "misses": self.misses,
        }
