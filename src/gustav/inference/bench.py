"""The benchmark.

This is what turns stage 2's bandwidth *estimate* into a measurement. The numbers it
writes (``bench.tokens_per_second``, ``bench.time_to_first_token_ms``,
``bench.vram_peak_mb``, tagged with the model and quantisation) are exactly the ones
``hw.speed.Measurements`` prefers, so a single run makes every later plan honest — and
gives the expansion speed floor something real to compare against.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from gustav.hw.speed import (
    METRIC_TOKENS_PER_SECOND,
    METRIC_TTFT_MS,
    METRIC_VRAM_PEAK_MB,
    Measurements,
)
from gustav.inference.backend import GenerationParams, Message

#: A short, fixed set: the point is comparability between runs, not coverage.
DEFAULT_PROMPTS: tuple[tuple[Message, ...], ...] = (
    ({"role": "user", "content": "Explain what a mutex protects, in two sentences."},),
    ({"role": "user", "content": "Write a Python function that reverses a linked list."},),
    (
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Summarise why floating point addition is not associative."},
    ),
)
#: A long prompt, to measure prefill rather than decode.
LONG_PROMPT: tuple[Message, ...] = (
    {
        "role": "user",
        "content": (
            "Here is a log file. Say in one line what went wrong.\n"
            + "\n".join(f"[{i:04d}] worker {i % 8} processed batch {i} in {i % 37} ms"
                        for i in range(400))
            + "\n[0401] worker 3 failed: connection reset by peer"
        ),
    },
)


@dataclass
class Sample:
    tokens_per_second: float
    time_to_first_token_ms: float
    prompt_tokens: int
    completion_tokens: int
    vram_peak_mb: float | None = None


@dataclass
class BenchResult:
    model: str
    quantization: str
    adapters: dict[str, str] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)
    duration_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def _median(self, attribute: str) -> float:
        values = [getattr(s, attribute) for s in self.samples if getattr(s, attribute)]
        return round(statistics.median(values), 2) if values else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self._median("tokens_per_second")

    @property
    def time_to_first_token_ms(self) -> float:
        return self._median("time_to_first_token_ms")

    @property
    def vram_peak_mb(self) -> float:
        values = [s.vram_peak_mb for s in self.samples if s.vram_peak_mb]
        return round(max(values), 1) if values else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.samples)

    def summary(self) -> str:
        return (
            f"{self.model} @ {self.quantization}: {self.tokens_per_second:.1f} tok/s, "
            f"{self.time_to_first_token_ms:.0f} ms to first token, "
            f"{self.vram_peak_mb / 1024:.2f} GiB peak VRAM "
            f"({len(self.samples)} samples, {self.total_tokens} tokens)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "samples"},
            "samples": [asdict(s) for s in self.samples],
            "tokens_per_second": self.tokens_per_second,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "vram_peak_mb": self.vram_peak_mb,
            "summary": self.summary(),
        }


async def run_benchmark(
    harness,
    *,
    prompts: Sequence[Sequence[Message]] = DEFAULT_PROMPTS,
    max_tokens: int = 96,
    repeats: int = 1,
    warmup: bool = True,
    include_long_prompt: bool = True,
    measurements: Measurements | None = None,
) -> BenchResult:
    """Measure speed, latency and peak VRAM, and record it for the planner."""
    described = harness.describe()
    result = BenchResult(
        model=described.get("base_model") or "unknown",
        quantization=described.get("quantization") or "unknown",
        adapters=dict(described.get("adapters") or {}),
    )
    # Greedy and fixed length: variance between runs should come from the machine,
    # not from sampling.
    params = GenerationParams(max_tokens=max_tokens, temperature=0.0, repetition_penalty=1.0)
    started = time.monotonic()

    if warmup:
        # The first generation pays for CUDA kernels and allocator warm-up.
        await harness.backend.generate(list(prompts[0]), params.merged(max_tokens=8))
        result.notes.append("one warm-up generation was discarded")

    cases = list(prompts) * max(repeats, 1)
    if include_long_prompt:
        cases.append(list(LONG_PROMPT))
    for case in cases:
        generation = await harness.backend.generate(list(case), params)
        result.samples.append(
            Sample(
                tokens_per_second=generation.tokens_per_second,
                time_to_first_token_ms=generation.time_to_first_token_ms,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                vram_peak_mb=generation.vram_peak_mb,
            )
        )
    result.duration_seconds = round(time.monotonic() - started, 2)

    if measurements is not None and result.samples:
        record(measurements, result)
    return result


def record(measurements: Measurements, result: BenchResult) -> None:
    """Write the measurement where the hardware planner will find it."""
    for metric, value in (
        (METRIC_TOKENS_PER_SECOND, result.tokens_per_second),
        (METRIC_TTFT_MS, result.time_to_first_token_ms),
        (METRIC_VRAM_PEAK_MB, result.vram_peak_mb),
    ):
        if value:
            measurements.record(
                metric,
                value,
                model_id=result.model,
                quant_id=result.quantization,
                adapters=sorted(result.adapters),
                samples=len(result.samples),
            )
