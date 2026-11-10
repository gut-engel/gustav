"""Speed estimation, and the measurements that replace it.

Token generation is memory-bandwidth bound: every token reads the weights it needs. So
the first-order estimate is ``bandwidth / bytes-read-per-token``, scaled by how much of
the theoretical bandwidth the framework actually achieves.

This is an *estimate*, and it is labelled as one everywhere it surfaces. Stage 3's
benchmark writes real numbers into the telemetry table and ``Measurements`` prefers
them from then on — which is also how the planner's speed floor becomes a measurement
rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from gustav.hw.ladder import ModelSpec, QuantSpec
from gustav.hw.memory import MIB, weights_mb
from gustav.hw.types import GpuInfo

#: Per-layer kernel-launch and Python overhead in a transformers generate() loop.
LAUNCH_MS_PER_LAYER = 0.04
#: Prefill processes the prompt in parallel, so it is far faster per token than decode.
PREFILL_SPEEDUP = 25.0
#: Used when NVML cannot tell us the bandwidth.
FALLBACK_BANDWIDTH_GB_S = 300.0

METRIC_TOKENS_PER_SECOND = "bench.tokens_per_second"
METRIC_TTFT_MS = "bench.time_to_first_token_ms"
METRIC_VRAM_PEAK_MB = "bench.vram_peak_mb"


@dataclass
class SpeedEstimate:
    tokens_per_second: float
    time_to_first_token_ms: float
    measured: bool
    basis: str

    def describe(self) -> str:
        kind = "measured" if self.measured else "estimated"
        return (
            f"{self.tokens_per_second:.0f} tok/s, "
            f"{self.time_to_first_token_ms:.0f} ms to first token ({kind}, {self.basis})"
        )


def bytes_read_per_token(
    model: ModelSpec,
    quant: QuantSpec,
    *,
    context_tokens: int,
    kv_dtype_bytes: float = 2.0,
) -> float:
    """What one decode step must pull out of VRAM.

    For a Mixture-of-Experts model only the active experts are read, which is exactly
    why a 30B MoE can decode at the speed of a 3B dense model.
    """
    if quant.linear_only:
        linear_bytes = model.active_linear_params * quant.bytes_per_param
        other_bytes = model.unquantized_params * 2.0
    else:
        linear_bytes = model.active_linear_params * quant.bytes_per_param
        other_bytes = model.unquantized_params * quant.bytes_per_param
    kv_bytes = model.kv_bytes_per_token(kv_dtype_bytes) * context_tokens
    return linear_bytes + other_bytes + kv_bytes


def estimate_speed(
    model: ModelSpec,
    quant: QuantSpec,
    gpu: GpuInfo | None,
    *,
    context_tokens: int = 1024,
    prompt_tokens: int = 512,
    kv_dtype_bytes: float = 2.0,
    speculative_factor: float = 1.0,
) -> SpeedEstimate:
    bandwidth = (gpu.bandwidth_gb_s if gpu and gpu.bandwidth_gb_s else FALLBACK_BANDWIDTH_GB_S)
    per_token_gb = bytes_read_per_token(
        model, quant, context_tokens=context_tokens, kv_dtype_bytes=kv_dtype_bytes
    ) / 1e9
    bandwidth_tps = bandwidth * quant.bandwidth_efficiency / max(per_token_gb, 1e-9)

    # Even an infinitely fast memory bus cannot beat the per-layer launch overhead.
    overhead_tps = 1000.0 / max(model.layers * LAUNCH_MS_PER_LAYER, 1e-6)
    tps = min(bandwidth_tps, overhead_tps) * speculative_factor

    prefill_tps = tps * PREFILL_SPEEDUP
    ttft = 1000.0 * prompt_tokens / max(prefill_tps, 1e-6)
    basis = (
        f"{bandwidth:.0f} GB/s x {quant.bandwidth_efficiency:.2f} / "
        f"{per_token_gb * 1000:.1f} MB per token"
    )
    if not gpu:
        basis = "no GPU: CPU inference, assume far slower than this"
        tps = min(tps, 3.0)
    return SpeedEstimate(
        tokens_per_second=round(tps, 1),
        time_to_first_token_ms=round(ttft, 1),
        measured=False,
        basis=basis,
    )


class Measurements:
    """Benchmark results, read from the telemetry table."""

    def __init__(self, db=None):
        self.db = db

    def _lookup(self, metric: str, model_id: str, quant_id: str) -> float | None:
        if self.db is None:
            return None
        row = self.db.query_one(
            "SELECT value FROM telemetry WHERE metric = ?"
            " AND json_extract(tags, '$.model') = ?"
            " AND json_extract(tags, '$.quant') = ?"
            " ORDER BY id DESC LIMIT 1",
            (metric, model_id, quant_id),
        )
        return float(row["value"]) if row else None

    def speed(
        self, model: ModelSpec, quant: QuantSpec, gpu: GpuInfo | None, **kwargs
    ) -> SpeedEstimate:
        """A measured speed if we have ever benchmarked this pair; else an estimate."""
        tps = self._lookup(METRIC_TOKENS_PER_SECOND, model.id, quant.id)
        if tps is None:
            return estimate_speed(model, quant, gpu, **kwargs)
        ttft = self._lookup(METRIC_TTFT_MS, model.id, quant.id)
        estimated = estimate_speed(model, quant, gpu, **kwargs)
        return SpeedEstimate(
            tokens_per_second=tps,
            time_to_first_token_ms=ttft if ttft is not None else estimated.time_to_first_token_ms,
            measured=True,
            basis="benchmarked on this machine",
        )

    def vram_peak_mb(self, model: ModelSpec, quant: QuantSpec) -> float | None:
        return self._lookup(METRIC_VRAM_PEAK_MB, model.id, quant.id)

    def record(
        self, metric: str, value: float, *, model_id: str, quant_id: str, **tags
    ) -> None:
        if self.db is None:
            return
        import json

        self.db.insert(
            "telemetry",
            {
                "metric": metric,
                "value": float(value),
                "tags": json.dumps({"model": model_id, "quant": quant_id, **tags}),
            },
        )
