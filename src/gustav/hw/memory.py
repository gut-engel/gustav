"""VRAM estimation.

Every term is named and explained, because the point of this module is to refuse a
configuration *before* it OOMs rather than to be clever. Estimates are deliberately a
little pessimistic: the cost of over-estimating is a smaller model, the cost of
under-estimating is a crash in the middle of the night.

The numbers that bite in practice and are usually left out of back-of-the-envelope
sizing are both here:

* **Unquantised weights.** bitsandbytes only touches ``nn.Linear``. For Qwen3's
  151 936-token vocabulary the embedding and lm_head alone are 2.5 GiB in 16-bit on the
  8B model — more than a third of its 4-bit footprint.
* **Training logits.** Cross-entropy over the whole sequence materialises a
  ``batch x seq x vocab`` tensor in fp32: 0.6 GiB at seq 1024 for that vocabulary, and
  roughly twice that at peak. It is often the single largest training allocation.

Stage 3 replaces estimates with measured peaks (see ``MeasurementStore``); until then
every estimate is labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gustav.hw.ladder import ModelSpec, QuantSpec

MIB = 1024 * 1024

#: CUDA context, cuBLAS/cuDNN workspaces and the bitsandbytes kernels, before any weights.
CUDA_CONTEXT_MB = 600
#: PyTorch's caching allocator never achieves perfect packing.
FRAGMENTATION = 1.08
#: Rough number of 16-bit temporaries live per token inside one transformer block.
ACTIVATION_TENSORS = 4
#: Peak logits allocation is about twice the logits tensor (upcast + softmax/loss).
LOGITS_PEAK_FACTOR = 2.0

KV_DTYPE_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp32": 4.0, "int8": 1.0, "fp8": 1.0}


def kv_dtype_bytes(name: str, *, fallback: float = 2.0) -> float:
    return KV_DTYPE_BYTES.get(name, fallback)


@dataclass
class MemoryEstimate:
    """A VRAM estimate, broken down so a refusal can say *why*."""

    label: str
    weights_mb: float = 0.0
    unquantized_weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    activations_mb: float = 0.0
    logits_mb: float = 0.0
    gradients_mb: float = 0.0
    optimizer_mb: float = 0.0
    adapters_mb: float = 0.0
    overhead_mb: float = float(CUDA_CONTEXT_MB)
    measured: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def subtotal_mb(self) -> float:
        return (
            self.weights_mb
            + self.unquantized_weights_mb
            + self.kv_cache_mb
            + self.activations_mb
            + self.logits_mb
            + self.gradients_mb
            + self.optimizer_mb
            + self.adapters_mb
            + self.overhead_mb
        )

    @property
    def total_mb(self) -> float:
        return round(self.subtotal_mb * FRAGMENTATION, 1)

    def breakdown(self) -> dict[str, float]:
        items = {
            "quantized weights": self.weights_mb,
            "16-bit weights (embeddings, lm_head, norms)": self.unquantized_weights_mb,
            "KV cache": self.kv_cache_mb,
            "activations": self.activations_mb,
            "logits + loss": self.logits_mb,
            "gradients": self.gradients_mb,
            "optimizer state": self.optimizer_mb,
            "adapters": self.adapters_mb,
            "CUDA context and workspaces": self.overhead_mb,
            f"fragmentation ({(FRAGMENTATION - 1) * 100:.0f}%)": self.total_mb - self.subtotal_mb,
        }
        return {k: round(v, 1) for k, v in items.items() if v > 0.05}

    def explain(self) -> list[str]:
        width = max(len(k) for k in self.breakdown()) if self.breakdown() else 0
        lines = [f"{self.label}: {self.total_mb / 1024:.2f} GiB"
                 + ("" if self.measured else " (estimated)")]
        lines += [f"  {k:<{width}}  {v / 1024:6.2f} GiB" for k, v in self.breakdown().items()]
        lines += [f"  ! {note}" for note in self.notes]
        return lines


def weights_mb(model: ModelSpec, quant: QuantSpec) -> tuple[float, float]:
    """(quantised, unquantised) weight footprint in MiB."""
    if quant.linear_only:
        quantized = model.linear_params * quant.bytes_per_param / MIB
        unquantized = model.unquantized_params * 2.0 / MIB
    else:
        quantized = 0.0
        unquantized = model.total_params * quant.bytes_per_param / MIB
    return round(quantized, 1), round(unquantized, 1)


def kv_cache_mb(
    model: ModelSpec, *, seq_len: int, batch: int = 1, dtype: str = "fp16"
) -> float:
    per_token = model.kv_bytes_per_token(kv_dtype_bytes(dtype))
    return round(per_token * seq_len * batch / MIB, 1)


def activations_mb(
    model: ModelSpec,
    *,
    seq_len: int,
    batch: int = 1,
    dtype_bytes: float = 2.0,
    training: bool = False,
    gradient_checkpointing: bool = True,
) -> float:
    """Live activation memory at peak.

    One block's temporaries are ``hidden`` plus the MLP's wider intermediate; with
    gradient checkpointing a training step additionally keeps one ``hidden``-sized
    tensor per layer boundary and recomputes the rest.
    """
    inter = model.moe_intermediate * max(model.moe_active_experts, 1) if model.is_moe else model.intermediate
    per_token = ACTIVATION_TENSORS * model.hidden + 3 * inter
    block_peak = per_token * seq_len * batch * dtype_bytes / MIB
    if not training:
        return round(block_peak, 1)
    if gradient_checkpointing:
        boundaries = model.layers * model.hidden * seq_len * batch * dtype_bytes / MIB
        return round(boundaries + block_peak, 1)
    return round(model.layers * block_peak, 1)


def logits_mb(
    model: ModelSpec, *, seq_len: int, batch: int = 1, dtype_bytes: float = 4.0,
    all_positions: bool = True,
) -> float:
    positions = seq_len if all_positions else 1
    raw = model.vocab * positions * batch * dtype_bytes / MIB
    return round(raw * LOGITS_PEAK_FACTOR, 1)


def inference_estimate(
    model: ModelSpec,
    quant: QuantSpec,
    *,
    seq_len: int,
    batch: int = 1,
    kv_dtype: str = "fp16",
    adapter_count: int = 0,
    lora_rank: int = 16,
    draft: tuple[ModelSpec, QuantSpec] | None = None,
) -> MemoryEstimate:
    quantized, unquantized = weights_mb(model, quant)
    estimate = MemoryEstimate(
        label=f"inference: {model.label} @ {quant.id}, seq {seq_len}, batch {batch}",
        weights_mb=quantized,
        unquantized_weights_mb=unquantized,
        kv_cache_mb=kv_cache_mb(model, seq_len=seq_len, batch=batch, dtype=kv_dtype),
        activations_mb=activations_mb(model, seq_len=seq_len, batch=batch),
        # Generation only needs the last position's logits.
        logits_mb=logits_mb(model, seq_len=seq_len, batch=batch, all_positions=False),
    )
    if adapter_count:
        # Adapters are held in 16-bit; several can be resident at once.
        params = model.lora_params(lora_rank) * adapter_count
        estimate.adapters_mb = round(params * 2 / MIB, 1)
    if draft is not None:
        draft_model, draft_quant = draft
        d_quant, d_unquant = weights_mb(draft_model, draft_quant)
        estimate.weights_mb += d_quant
        estimate.unquantized_weights_mb += d_unquant
        estimate.kv_cache_mb += kv_cache_mb(
            draft_model, seq_len=seq_len, batch=batch, dtype=kv_dtype
        )
        estimate.notes.append(
            f"includes the {draft_model.label} draft model for speculative decoding"
        )
    if model.is_moe and model.expert_offload:
        estimate.notes.append(
            "assumes all experts resident; expert offloading to RAM lowers this and costs speed"
        )
    return estimate


def training_estimate(
    model: ModelSpec,
    quant: QuantSpec,
    *,
    seq_len: int,
    batch: int = 1,
    lora_rank: int = 16,
    lora_targets: tuple[str, ...] | None = None,
    optimizer: str = "paged_adamw_8bit",
    gradient_checkpointing: bool = True,
    kv_dtype: str = "fp16",
) -> MemoryEstimate:
    """QLoRA training: the base model is frozen, only the adapter is trained."""
    quantized, unquantized = weights_mb(model, quant)
    trainable = model.lora_params(lora_rank, lora_targets)

    # Optimizer state per trainable parameter, in bytes.
    state_bytes = {
        "paged_adamw_8bit": 2.0,   # two 8-bit moments
        "adamw_8bit": 2.0,
        "adamw_torch": 8.0,        # two fp32 moments
        "adamw_bnb_8bit": 2.0,
        "sgd": 4.0,
        "adafactor": 4.0,
    }.get(optimizer, 8.0)

    estimate = MemoryEstimate(
        label=(
            f"training: {model.label} @ {quant.id}, LoRA r{lora_rank}, "
            f"seq {seq_len}, batch {batch}"
        ),
        weights_mb=quantized,
        unquantized_weights_mb=unquantized,
        # Training does not keep a generation KV cache, but attention still needs
        # the current window; counted inside activations.
        kv_cache_mb=0.0,
        activations_mb=activations_mb(
            model, seq_len=seq_len, batch=batch, training=True,
            gradient_checkpointing=gradient_checkpointing,
        ),
        logits_mb=logits_mb(model, seq_len=seq_len, batch=batch, all_positions=True),
        # Adapter weights are fp32 masters plus a 16-bit working copy.
        adapters_mb=round(trainable * 6 / MIB, 1),
        gradients_mb=round(trainable * 4 / MIB, 1),
        optimizer_mb=round(trainable * state_bytes / MIB, 1),
    )
    estimate.notes.append(
        f"{trainable / 1e6:.1f}M trainable parameters "
        f"({100 * trainable / model.total_params:.2f}% of the model)"
    )
    if not gradient_checkpointing:
        estimate.notes.append("gradient checkpointing is OFF: activations dominate")
    if not quant.trainable:
        estimate.notes.append(f"{quant.id} is not supported for training")
    return estimate


def ram_estimate(
    model: ModelSpec,
    quant: QuantSpec,
    *,
    warm_cache_mb: int = 0,
    expert_offload_mb: float = 0.0,
) -> float:
    """System RAM needed beside VRAM: the checkpoint is staged through host memory."""
    quantized, unquantized = weights_mb(model, quant)
    staging = (quantized + unquantized) * 0.5  # safetensors mmap + conversion buffers
    return round(staging + warm_cache_mb + expert_offload_mb + 1024, 1)
