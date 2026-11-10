"""Hardware detection.

NVML is the source of truth when it is there (it works without torch installed, which
matters: stages 1-2 have no ML dependencies). Everything is behind a ``Detector``
protocol so the planner can be tested against simulated machines — a 4 GB laptop, a
12 GB desktop, a four-card server — without owning any of them.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from gustav.hw.types import MIB, ForeignProcess, GpuInfo, Hardware, HostInfo

#: GPUs whose NVML-reported clock/bus width give the wrong answer, or that do not
#: report them at all. Keyed by a substring of the device name.
BANDWIDTH_OVERRIDES_GB_S: dict[str, float] = {
    "A100-SXM4-80GB": 2039.0,
    "A100-SXM4-40GB": 1555.0,
    "A100-PCIE": 1935.0,
    "H100": 3350.0,
    "H200": 4800.0,
}


def flash_attention_available() -> bool:
    """Whether the ``flash_attn`` package is importable.

    The GPU supporting FlashAttention-2 is not the same as having it installed, and
    promising it in the plan when it is missing makes the plan a lie. It is an optional
    extra: without it the backend uses PyTorch SDPA, which is close for small models.
    """
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


@runtime_checkable
class Detector(Protocol):
    def detect(self) -> Hardware: ...


def _host_info(data_dir: Path | None = None) -> HostInfo:
    import psutil

    memory = psutil.virtual_memory()
    target = data_dir or Path.cwd()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return HostInfo(
        cpu_count=os.cpu_count() or psutil.cpu_count(logical=True) or 1,
        cpu_model=_cpu_model(),
        ram_total_mb=memory.total // MIB,
        ram_available_mb=memory.available // MIB,
        disk_total_mb=usage.total // MIB,
        disk_free_mb=usage.free // MIB,
        platform=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
    )


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _bandwidth(name: str, bus_bits: int | None, clock_mhz: int | None) -> float | None:
    for needle, value in BANDWIDTH_OVERRIDES_GB_S.items():
        if needle.lower() in name.lower():
            return value
    if not bus_bits or not clock_mhz:
        return None
    # GDDR/HBM are double data rate: bytes/s = bus_bytes * clock * 2.
    return round(bus_bits / 8 * clock_mhz * 2 / 1000, 1)


class NvmlDetector:
    """Reads the NVIDIA management library. No torch, no CUDA context."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir

    def detect(self) -> Hardware:
        host = _host_info(self.data_dir)
        try:
            import pynvml as nvml
        except ImportError:
            return Hardware(
                gpus=(), host=host, source="none",
                notes=("nvidia-ml-py is not installed, so no GPU could be detected",),
            )
        try:
            nvml.nvmlInit()
        except Exception as exc:  # noqa: BLE001 - no driver, no GPU, a container without /dev/nvidia*
            return Hardware(
                gpus=(), host=host, source="none",
                notes=(f"NVML unavailable ({type(exc).__name__}): running CPU-only",),
            )
        try:
            return Hardware(
                gpus=tuple(self._gpu(nvml, i) for i in range(nvml.nvmlDeviceGetCount())),
                host=host,
                source="nvml",
                driver_version=_text(nvml.nvmlSystemGetDriverVersion()),
                cuda_version=_cuda_version(nvml),
            )
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def _gpu(self, nvml, index: int) -> GpuInfo:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        name = _text(nvml.nvmlDeviceGetName(handle))
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        bus = _try(nvml.nvmlDeviceGetMemoryBusWidth, handle)
        clock = _try(nvml.nvmlDeviceGetMaxClockInfo, handle, getattr(nvml, "NVML_CLOCK_MEM", 2))
        return GpuInfo(
            index=index,
            name=name,
            uuid=_text(_try(nvml.nvmlDeviceGetUUID, handle) or ""),
            total_mb=memory.total // MIB,
            free_mb=memory.free // MIB,
            compute_capability=tuple(nvml.nvmlDeviceGetCudaComputeCapability(handle)),
            memory_bus_bits=bus,
            memory_clock_mhz=clock,
            bandwidth_gb_s=_bandwidth(name, bus, clock),
            cuda_cores=_try(nvml.nvmlDeviceGetNumGpuCores, handle),
            temperature_c=_try(
                nvml.nvmlDeviceGetTemperature, handle, getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
            ),
            power_limit_w=_scale(_try(nvml.nvmlDeviceGetEnforcedPowerLimit, handle), 1000),
            utilization_pct=_utilization(nvml, handle),
            foreign_processes=self._foreign(nvml, handle),
        )

    def _foreign(self, nvml, handle) -> tuple[ForeignProcess, ...]:
        """Processes other than us that hold VRAM on this card."""
        out: list[ForeignProcess] = []
        ours = {os.getpid(), os.getppid()}
        for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                processes = getattr(nvml, getter)(handle)
            except Exception:  # noqa: BLE001
                continue
            for proc in processes:
                pid = int(getattr(proc, "pid", 0) or 0)
                if not pid or pid in ours:
                    continue
                used = getattr(proc, "usedGpuMemory", None)
                out.append(
                    ForeignProcess(
                        pid=pid,
                        name=_process_name(pid),
                        used_mb=(used // MIB) if isinstance(used, int) else 0,
                    )
                )
        return tuple({p.pid: p for p in out}.values())

    def detect_temperatures(self) -> dict[int, float]:
        """Cheap poll used by the scheduler's GPU-temperature guard."""
        try:
            import pynvml as nvml

            nvml.nvmlInit()
        except Exception:  # noqa: BLE001
            return {}
        try:
            out: dict[int, float] = {}
            for index in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                value = _try(
                    nvml.nvmlDeviceGetTemperature, handle,
                    getattr(nvml, "NVML_TEMPERATURE_GPU", 0),
                )
                if value is not None:
                    out[index] = float(value)
            return out
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


class SimulatedDetector:
    """A machine described in a dict. Used by tests and by ``--simulate``."""

    def __init__(self, hardware: Hardware):
        self.hardware = hardware

    def detect(self) -> Hardware:
        return self.hardware

    @classmethod
    def from_spec(
        cls,
        *,
        gpus: list[dict] | None = None,
        ram_gb: int = 64,
        cpu_count: int = 16,
        disk_free_gb: int = 500,
        name: str = "simulated",
    ) -> "SimulatedDetector":
        built = []
        for index, spec in enumerate(gpus or []):
            total_mb = int(spec.get("total_mb") or spec.get("vram_gb", 12) * 1024)
            cc = tuple(spec.get("compute_capability", (8, 9)))
            gpu_name = spec.get("name", f"Simulated GPU {cc[0]}.{cc[1]}")
            built.append(
                GpuInfo(
                    index=index,
                    name=gpu_name,
                    total_mb=total_mb,
                    free_mb=int(spec.get("free_mb", total_mb)),
                    compute_capability=cc,  # type: ignore[arg-type]
                    bandwidth_gb_s=spec.get("bandwidth_gb_s", 500.0),
                    foreign_processes=tuple(
                        ForeignProcess(**p) for p in spec.get("foreign_processes", [])
                    ),
                )
            )
        host = HostInfo(
            cpu_count=cpu_count,
            cpu_model="simulated",
            ram_total_mb=ram_gb * 1024,
            ram_available_mb=int(ram_gb * 1024 * 0.9),
            disk_total_mb=disk_free_gb * 1024 * 2,
            disk_free_mb=disk_free_gb * 1024,
            platform=name,
            python_version=sys.version.split()[0],
        )
        return cls(Hardware(gpus=tuple(built), host=host, source="simulated"))


#: Named machines used in tests and documentation.
PRESETS: dict[str, dict] = {
    "tiny-4gb": {
        "gpus": [{"name": "NVIDIA GeForce GTX 1650", "vram_gb": 4,
                  "compute_capability": (7, 5), "bandwidth_gb_s": 192.0}],
        "ram_gb": 16, "cpu_count": 6, "disk_free_gb": 100,
    },
    "small-12gb": {
        "gpus": [{"name": "NVIDIA GeForce RTX 4070 SUPER", "vram_gb": 12,
                  "compute_capability": (8, 9), "bandwidth_gb_s": 504.0}],
        "ram_gb": 64, "cpu_count": 24, "disk_free_gb": 1700,
    },
    "server-multigpu": {
        "gpus": [
            {"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24,
             "compute_capability": (8, 6), "bandwidth_gb_s": 936.0}
            for _ in range(4)
        ],
        "ram_gb": 256, "cpu_count": 64, "disk_free_gb": 8000,
    },
    "cpu-only": {"gpus": [], "ram_gb": 32, "cpu_count": 8, "disk_free_gb": 200},
}


def simulated(preset: str) -> SimulatedDetector:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; known: {', '.join(sorted(PRESETS))}")
    return SimulatedDetector.from_spec(name=preset, **PRESETS[preset])


def default_detector(data_dir: Path | None = None) -> Detector:
    """NVML normally; a simulated machine when ``GUSTAV_SIMULATE_HARDWARE`` is set."""
    preset = os.environ.get("GUSTAV_SIMULATE_HARDWARE")
    if preset:
        return simulated(preset)
    return NvmlDetector(data_dir)


def detect(data_dir: Path | None = None) -> Hardware:
    return default_detector(data_dir).detect()


# --- small helpers --------------------------------------------------------
def _try(func, *args):
    try:
        return func(*args)
    except Exception:  # noqa: BLE001 - every NVML call is optional
        return None


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value) if value is not None else ""


def _scale(value, divisor: float):
    return round(value / divisor, 1) if isinstance(value, (int, float)) else None


def _utilization(nvml, handle) -> float | None:
    rates = _try(nvml.nvmlDeviceGetUtilizationRates, handle)
    return float(rates.gpu) if rates is not None else None


def _cuda_version(nvml) -> str:
    raw = _try(nvml.nvmlSystemGetCudaDriverVersion_v2) or _try(nvml.nvmlSystemGetCudaDriverVersion)
    if not isinstance(raw, int):
        return ""
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _process_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001
        return f"pid-{pid}"
