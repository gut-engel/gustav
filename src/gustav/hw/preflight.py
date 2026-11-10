"""The pre-flight check.

Nothing loads a model without passing through here first. A refusal is a sentence a
human can act on ("needs 9.2 GiB, 8.1 GiB available after the 15% reserve; try
max_seq_len 1024 or the 4B model"), not a CUDA stack trace at three in the morning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gustav.hw.memory import MemoryEstimate
from gustav.hw.types import GpuInfo, Hardware


@dataclass(frozen=True)
class VramBudget:
    """How much of a card we are allowed to plan with."""

    gpu_index: int
    total_mb: int
    reserved_mb: int
    foreign_mb: int
    usable_mb: int
    reserve_fraction: float

    def describe(self) -> str:
        return (
            f"GPU{self.gpu_index}: {self.usable_mb / 1024:.2f} GiB usable of "
            f"{self.total_mb / 1024:.2f} GiB "
            f"({self.reserve_fraction:.0%} reserved for other software"
            + (f", {self.foreign_mb} MiB already in use by others" if self.foreign_mb else "")
            + ")"
        )


@dataclass
class PreflightResult:
    ok: bool
    estimate: MemoryEstimate
    budget: VramBudget | None
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def headroom_mb(self) -> float:
        if self.budget is None:
            return 0.0
        return round(self.budget.usable_mb - self.estimate.total_mb, 1)

    @property
    def utilisation(self) -> float:
        if self.budget is None or not self.budget.total_mb:
            return 0.0
        return round(self.estimate.total_mb / self.budget.total_mb, 3)

    def explain(self) -> list[str]:
        lines = list(self.estimate.explain())
        if self.budget:
            lines.append(f"  {self.budget.describe()}")
            lines.append(
                f"  headroom: {self.headroom_mb / 1024:+.2f} GiB "
                f"({self.utilisation:.0%} of the card at peak)"
            )
        lines += [f"  REFUSED: {m}" for m in self.blocking]
        lines += [f"  warning: {m}" for m in self.warnings]
        return lines

    def raise_if_blocked(self) -> None:
        if not self.ok:
            raise MemoryBudgetExceeded("\n".join(self.explain()))


class MemoryBudgetExceeded(RuntimeError):
    """Raised instead of letting CUDA OOM at an inconvenient moment."""


def vram_budget(
    gpu: GpuInfo,
    *,
    reserve_fraction: float = 0.15,
    min_reserve_mb: int = 0,
    count_foreign: bool = True,
) -> VramBudget:
    """Usable VRAM on one card.

    The reserve exists precisely to cover other software, so we take the larger of the
    configured reserve and what other processes are *actually* holding rather than
    subtracting both.
    """
    foreign = gpu.foreign_mb if count_foreign else 0
    reserve = max(int(gpu.total_mb * reserve_fraction), min_reserve_mb)
    reserved = max(reserve, foreign)
    return VramBudget(
        gpu_index=gpu.index,
        total_mb=gpu.total_mb,
        reserved_mb=reserved,
        foreign_mb=foreign,
        usable_mb=max(gpu.total_mb - reserved, 0),
        reserve_fraction=reserve_fraction,
    )


def check_vram(
    estimate: MemoryEstimate,
    budget: VramBudget | None,
    *,
    warn_fraction: float = 0.90,
    suggestions: list[str] | None = None,
) -> PreflightResult:
    result = PreflightResult(ok=True, estimate=estimate, budget=budget)
    if budget is None:
        result.ok = False
        result.blocking.append("no GPU available, so nothing can be loaded onto one")
        return result

    needed = estimate.total_mb
    if needed > budget.usable_mb:
        result.ok = False
        short = needed - budget.usable_mb
        message = (
            f"needs {needed / 1024:.2f} GiB but only {budget.usable_mb / 1024:.2f} GiB is "
            f"usable on GPU{budget.gpu_index} "
            f"({short / 1024:.2f} GiB short)"
        )
        if suggestions:
            message += "; try " + ", or ".join(suggestions)
        result.blocking.append(message)
    elif needed > budget.usable_mb * warn_fraction:
        result.warnings.append(
            f"only {(budget.usable_mb - needed) / 1024:.2f} GiB spare: a longer context or "
            f"another resident adapter will not fit"
        )
    return result


def check_ram(
    needed_mb: float,
    hardware: Hardware,
    *,
    reserve_mb: int = 8192,
) -> tuple[bool, str]:
    available = hardware.host.ram_total_mb - reserve_mb
    if needed_mb > available:
        return False, (
            f"needs {needed_mb / 1024:.1f} GiB of system RAM but only "
            f"{available / 1024:.1f} GiB is available after the "
            f"{reserve_mb / 1024:.0f} GiB reserve"
        )
    return True, ""


def check_disk(needed_mb: float, hardware: Hardware, *, reserve_mb: int = 10240) -> tuple[bool, str]:
    free = hardware.host.disk_free_mb - reserve_mb
    if needed_mb > free:
        return False, (
            f"needs {needed_mb / 1024:.1f} GiB of free disk but only {free / 1024:.1f} GiB "
            f"is available after the {reserve_mb / 1024:.0f} GiB reserve"
        )
    return True, ""
