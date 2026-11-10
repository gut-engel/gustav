"""What we know about the machine we are running on."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

MIB = 1024 * 1024


@dataclass(frozen=True)
class ForeignProcess:
    """Something else is using this GPU. Its VRAM is not ours to plan with."""

    pid: int
    name: str
    used_mb: int


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    compute_capability: tuple[int, int]
    uuid: str = ""
    #: Derived from bus width x memory clock when NVML reports them.
    bandwidth_gb_s: float | None = None
    memory_bus_bits: int | None = None
    memory_clock_mhz: int | None = None
    cuda_cores: int | None = None
    temperature_c: float | None = None
    power_limit_w: float | None = None
    utilization_pct: float | None = None
    foreign_processes: tuple[ForeignProcess, ...] = ()

    # --- capabilities, derived from the compute capability ---------------
    @property
    def supports_bf16(self) -> bool:
        """Ampere and newer. Turing must use fp16, which needs loss scaling."""
        return self.compute_capability >= (8, 0)

    @property
    def supports_fp8(self) -> bool:
        return self.compute_capability >= (8, 9)

    @property
    def has_tensor_cores(self) -> bool:
        return self.compute_capability >= (7, 0)

    @property
    def supports_flash_attention_2(self) -> bool:
        """FA2 needs sm_80+. Everything older falls back to PyTorch SDPA."""
        return self.compute_capability >= (8, 0)

    @property
    def attention_backend(self) -> str:
        return "flash_attention_2" if self.supports_flash_attention_2 else "sdpa"

    @property
    def preferred_dtype(self) -> str:
        return "bfloat16" if self.supports_bf16 else "float16"

    @property
    def architecture(self) -> str:
        return {
            (6, 0): "Pascal", (6, 1): "Pascal",
            (7, 0): "Volta", (7, 5): "Turing",
            (8, 0): "Ampere", (8, 6): "Ampere", (8, 7): "Ampere", (8, 9): "Ada Lovelace",
            (9, 0): "Hopper", (10, 0): "Blackwell", (12, 0): "Blackwell",
        }.get(self.compute_capability, f"sm_{self.compute_capability[0]}{self.compute_capability[1]}")

    @property
    def foreign_mb(self) -> int:
        return sum(p.used_mb for p in self.foreign_processes)

    def describe(self) -> str:
        bw = f", ~{self.bandwidth_gb_s:.0f} GB/s" if self.bandwidth_gb_s else ""
        return (
            f"GPU{self.index} {self.name}: {self.total_mb / 1024:.1f} GiB, "
            f"{self.architecture} (sm_{self.compute_capability[0]}{self.compute_capability[1]})"
            f"{bw}, {self.preferred_dtype}, {self.attention_backend}"
        )


@dataclass(frozen=True)
class HostInfo:
    cpu_count: int
    cpu_model: str
    ram_total_mb: int
    ram_available_mb: int
    disk_total_mb: int
    disk_free_mb: int
    platform: str = ""
    python_version: str = ""


@dataclass(frozen=True)
class Hardware:
    gpus: tuple[GpuInfo, ...]
    host: HostInfo
    #: "nvml", "simulated", or "none" when there is no usable accelerator.
    source: str = "nvml"
    driver_version: str = ""
    cuda_version: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    notes: tuple[str, ...] = ()

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_mb(self) -> int:
        return sum(g.total_mb for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    def gpu(self, index: int) -> GpuInfo:
        for candidate in self.gpus:
            if candidate.index == index:
                return candidate
        raise KeyError(f"no GPU with index {index}")

    def fingerprint(self) -> str:
        """A stable identity for this machine's *capacity*.

        Deliberately ignores anything that moves on its own — free memory, clocks,
        temperature, what else is running — so it changes when the hardware changes and
        not when the desktop opens a browser tab.
        """
        payload = {
            "gpus": [
                {
                    "name": g.name,
                    "total_mb": g.total_mb,
                    "cc": list(g.compute_capability),
                }
                for g in sorted(self.gpus, key=lambda g: (g.name, g.total_mb, g.index))
            ],
            # Rounded to GiB: a few MiB of difference is not a new machine.
            "ram_gb": round(self.host.ram_total_mb / 1024),
            "cpu_count": self.host.cpu_count,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint()
        for gpu in data["gpus"]:
            gpu["compute_capability"] = list(gpu["compute_capability"])
        return data

    def describe(self) -> list[str]:
        lines = [g.describe() for g in self.gpus]
        if not self.gpus:
            lines.append("no GPU detected: inference will be slow and training is off")
        for gpu in self.gpus:
            if gpu.foreign_processes:
                procs = ", ".join(f"{p.name}({p.pid}) {p.used_mb} MiB" for p in gpu.foreign_processes)
                lines.append(f"GPU{gpu.index} shared with: {procs}")
        lines.append(
            f"host: {self.host.cpu_count} cores, {self.host.ram_total_mb / 1024:.0f} GiB RAM "
            f"({self.host.ram_available_mb / 1024:.0f} GiB free), "
            f"{self.host.disk_free_mb / 1024:.0f} GiB disk free"
        )
        lines.extend(self.notes)
        return lines
