from __future__ import annotations

import pytest

from gustav.hw.detect import (
    BANDWIDTH_OVERRIDES_GB_S,
    NvmlDetector,
    PRESETS,
    SimulatedDetector,
    _bandwidth,
    default_detector,
    simulated,
)
from gustav.hw.types import ForeignProcess, GpuInfo, Hardware, HostInfo


def gpu(**kwargs) -> GpuInfo:
    base = dict(index=0, name="Test GPU", total_mb=12288, free_mb=12000,
                compute_capability=(8, 9))
    base.update(kwargs)
    return GpuInfo(**base)


# --- capability derivation ------------------------------------------------
@pytest.mark.parametrize(
    "cc,bf16,fa2,dtype,arch",
    [
        ((7, 5), False, False, "float16", "Turing"),
        ((8, 0), True, True, "bfloat16", "Ampere"),
        ((8, 6), True, True, "bfloat16", "Ampere"),
        ((8, 9), True, True, "bfloat16", "Ada Lovelace"),
        ((9, 0), True, True, "bfloat16", "Hopper"),
    ],
)
def test_capabilities_follow_compute_capability(cc, bf16, fa2, dtype, arch):
    card = gpu(compute_capability=cc)
    assert card.supports_bf16 is bf16
    assert card.supports_flash_attention_2 is fa2
    assert card.preferred_dtype == dtype
    assert card.architecture == arch
    assert card.attention_backend == ("flash_attention_2" if fa2 else "sdpa")


def test_turing_gets_no_bf16_and_no_flash_attention():
    """The 4 GB profile depends on this: a GTX 1650 must not be handed bf16 or FA2."""
    card = gpu(name="NVIDIA GeForce GTX 1650", compute_capability=(7, 5), total_mb=4096)
    assert not card.supports_bf16 and not card.supports_flash_attention_2
    assert card.preferred_dtype == "float16" and card.attention_backend == "sdpa"


def test_fp8_only_on_ada_and_newer():
    assert gpu(compute_capability=(8, 9)).supports_fp8
    assert not gpu(compute_capability=(8, 6)).supports_fp8


# --- bandwidth ------------------------------------------------------------
def test_bandwidth_is_derived_from_bus_width_and_clock():
    # The RTX 4070 SUPER: 192-bit bus at 10501 MHz -> 504 GB/s, its published figure.
    assert _bandwidth("RTX 4070 SUPER", 192, 10501) == pytest.approx(504.0, abs=0.5)


def test_bandwidth_overrides_win_for_known_cards():
    name = next(iter(BANDWIDTH_OVERRIDES_GB_S))
    assert _bandwidth(name, 192, 10501) == BANDWIDTH_OVERRIDES_GB_S[name]


def test_bandwidth_is_none_when_nvml_is_silent():
    assert _bandwidth("Mystery GPU", None, None) is None


# --- foreign processes ----------------------------------------------------
def test_foreign_vram_is_summed():
    card = gpu(foreign_processes=(
        ForeignProcess(1, "plasmashell", 163),
        ForeignProcess(2, "zen", 152),
    ))
    assert card.foreign_mb == 315
    assert "plasmashell" in Hardware(gpus=(card,), host=_host()).describe()[1]


def _host() -> HostInfo:
    return HostInfo(cpu_count=8, cpu_model="x", ram_total_mb=32768, ram_available_mb=20000,
                    disk_total_mb=500000, disk_free_mb=200000)


# --- fingerprint ----------------------------------------------------------
def test_fingerprint_is_stable_across_moving_values():
    a = Hardware(gpus=(gpu(free_mb=12000, temperature_c=40, utilization_pct=5),), host=_host())
    b = Hardware(
        gpus=(gpu(free_mb=3000, temperature_c=81, utilization_pct=99,
                  foreign_processes=(ForeignProcess(9, "game", 4000),)),),
        host=_host(),
    )
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_when_a_gpu_is_added():
    one = Hardware(gpus=(gpu(),), host=_host())
    two = Hardware(gpus=(gpu(), gpu(index=1)), host=_host())
    assert one.fingerprint() != two.fingerprint()


def test_fingerprint_changes_when_vram_changes():
    small = Hardware(gpus=(gpu(total_mb=8192),), host=_host())
    big = Hardware(gpus=(gpu(total_mb=24576),), host=_host())
    assert small.fingerprint() != big.fingerprint()


def test_fingerprint_changes_when_ram_changes_by_a_lot():
    before = Hardware(gpus=(gpu(),), host=_host())
    after = Hardware(
        gpus=(gpu(),),
        host=HostInfo(cpu_count=8, cpu_model="x", ram_total_mb=131072, ram_available_mb=120000,
                      disk_total_mb=500000, disk_free_mb=200000),
    )
    assert before.fingerprint() != after.fingerprint()


def test_fingerprint_ignores_a_few_mib_of_ram():
    before = Hardware(gpus=(gpu(),), host=_host())
    after = Hardware(
        gpus=(gpu(),),
        host=HostInfo(cpu_count=8, cpu_model="x", ram_total_mb=32768 + 50,
                      ram_available_mb=20000, disk_total_mb=500000, disk_free_mb=200000),
    )
    assert before.fingerprint() == after.fingerprint()


def test_gpu_order_does_not_change_the_fingerprint():
    a = Hardware(gpus=(gpu(index=0, name="A"), gpu(index=1, name="B")), host=_host())
    b = Hardware(gpus=(gpu(index=0, name="B"), gpu(index=1, name="A")), host=_host())
    assert a.fingerprint() == b.fingerprint()


# --- presets and simulation ----------------------------------------------
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_is_describable(preset):
    hardware = simulated(preset).detect()
    assert hardware.source == "simulated"
    assert hardware.describe()
    assert hardware.fingerprint()


def test_preset_shapes():
    assert simulated("tiny-4gb").detect().gpus[0].total_mb == 4096
    assert simulated("small-12gb").detect().gpus[0].total_mb == 12288
    assert simulated("server-multigpu").detect().gpu_count == 4
    assert not simulated("cpu-only").detect().has_gpu


def test_unknown_preset_lists_the_known_ones():
    with pytest.raises(KeyError, match="tiny-4gb"):
        simulated("nope")


def test_env_var_switches_to_a_simulated_machine(monkeypatch):
    monkeypatch.setenv("GUSTAV_SIMULATE_HARDWARE", "server-multigpu")
    assert default_detector().detect().gpu_count == 4


def test_simulated_detector_carries_foreign_processes():
    detector = SimulatedDetector.from_spec(
        gpus=[{"vram_gb": 12, "foreign_processes": [{"pid": 1, "name": "kwin", "used_mb": 200}]}]
    )
    assert detector.detect().gpus[0].foreign_mb == 200


# --- the real machine -----------------------------------------------------
def test_detection_never_raises_even_without_a_driver(monkeypatch, tmp_path):
    """A container with no /dev/nvidia* must degrade, not crash."""
    import builtins

    real_import = builtins.__import__

    def no_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no pynvml here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pynvml)
    hardware = NvmlDetector(tmp_path).detect()
    assert not hardware.has_gpu
    assert hardware.source == "none"
    assert "nvidia-ml-py" in hardware.notes[0]
    assert hardware.host.cpu_count >= 1


def test_host_info_is_real(tmp_path):
    hardware = NvmlDetector(tmp_path).detect()
    assert hardware.host.ram_total_mb > 0
    assert hardware.host.disk_free_mb > 0
    assert hardware.host.python_version.startswith("3.")


def test_missing_gpu_index_is_an_error():
    with pytest.raises(KeyError):
        Hardware(gpus=(gpu(),), host=_host()).gpu(7)
