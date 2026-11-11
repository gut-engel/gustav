"""LoRA training, in a child process, on the card the plan reserved for it.

The loop is written out rather than handed to a framework trainer, because the things
that make an *always-on* trainer different from a script are all in the loop:

* **it can be interrupted.** ``should_stop`` is checked every step; a user message mid
  training means save the adapter, report where it got to, and give the GPU back.
* **it checkpoints.** The adapter is written every ``checkpoint_every_steps``, so a
  reboot resumes the night instead of restarting it.
* **it has ceilings.** ``max_steps`` and ``max_seconds``, whichever comes first,
  because an unbounded night is a machine that is hot at breakfast.
* **it weights examples.** A per-sample loss scale is the whole point of the dataset
  mix, and a framework's label-smoothing path would quietly average it away.

What it does *not* do is decide anything about promotion. It writes a candidate version
into a staging directory and returns; the gate measures it, and the guard is the only
thing that may make it live.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gustav.config.schema import TrainingConfig
from gustav.hw.ladder import ModelSpec, QuantSpec
from gustav.hw.planner import Plan
from gustav.train.dataset import Mix, Sample, encode, normalised_weights

Progress = Callable[[str, dict[str, Any]], None]


class TrainingError(RuntimeError):
    pass


@dataclass
class TrainOutcome:
    module: str
    examples: int = 0
    steps: int = 0
    loss_start: float | None = None
    loss_end: float | None = None
    seconds: float = 0.0
    path: Path | None = None
    preempted: bool = False
    stopped_by: str = ""
    trainable_params: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def trained(self) -> bool:
        return self.steps > 0 and self.path is not None

    def describe(self) -> str:
        if not self.trained:
            return f"{self.module}: nothing trained ({self.stopped_by or 'no data'})"
        delta = (
            f", loss {self.loss_start:.3f} -> {self.loss_end:.3f}"
            if self.loss_start is not None and self.loss_end is not None
            else ""
        )
        return (
            f"{self.module}: {self.steps} steps on {self.examples} examples in "
            f"{self.seconds:.0f}s{delta}"
            + (" (preempted)" if self.preempted else "")
        )


def _lr_at(step: int, total: int, *, base: float, warmup: int) -> float:
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _collate(batch: Sequence[tuple[list[int], list[int], float]], pad_id: int, torch: Any):
    width = max(len(ids) for ids, _, _ in batch)
    input_ids, labels, mask, weights = [], [], [], []
    for ids, lab, weight in batch:
        padding = width - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        labels.append(lab + [-100] * padding)
        mask.append([1] * len(ids) + [0] * padding)
        weights.append(weight)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


def train_adapter(
    *,
    mix: Mix,
    plan: Plan,
    spec: ModelSpec,
    quant: QuantSpec,
    models_dir: Path,
    output_dir: Path,
    config: TrainingConfig,
    on_progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
    resume_step: int = 0,
    seed: int = 1734,
) -> TrainOutcome:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gustav.inference.transformers_backend import quantization_config, resolve_model_path

    outcome = TrainOutcome(module=mix.module)
    if not mix.samples:
        outcome.stopped_by = "no examples"
        return outcome

    def report(message: str, **data: Any) -> None:
        if on_progress is not None:
            on_progress(message, data)

    stop = should_stop or (lambda: False)
    started = time.monotonic()
    path = resolve_model_path(models_dir, spec)
    device_index = (plan.training_gpus or plan.inference_gpus or [0])[0]
    torch.cuda.set_device(device_index)
    torch.manual_seed(seed)

    report(f"loading {spec.label} for training on GPU {device_index}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        plan.compute_dtype, torch.float16
    )
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": {"": device_index},
    }
    quant_config = quantization_config(quant, dtype)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)

    # --- encode before touching the optimiser --------------------------
    seq_len = plan.train_seq_len or 1024
    encoded: list[tuple[list[int], list[int], float]] = []
    weights = normalised_weights(mix.samples)
    skipped = 0
    for sample, weight in zip(mix.samples, weights, strict=True):
        pair = encode(tokenizer, sample, max_len=seq_len)
        if pair is None:
            skipped += 1
            continue
        encoded.append((pair[0], pair[1], weight))
    if not encoded:
        outcome.stopped_by = "every example was empty after encoding"
        del model
        torch.cuda.empty_cache()
        return outcome
    outcome.examples = len(encoded)
    outcome.detail["skipped_examples"] = skipped

    if quant_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=plan.gradient_checkpointing
        )
    if plan.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    rank = config.rank or plan.lora_rank
    lora = LoraConfig(
        r=rank,
        lora_alpha=int(rank * config.alpha_ratio),
        lora_dropout=config.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(spec.lora_targets),
    )
    # Picking the night back up: the adapter on disk is the state, the step index is how
    # far through the data it got. If the weights are *not* there — crash recovery
    # restored the tree from a snapshot, say — then there is nothing to resume, and
    # starting at step N with a fresh adapter would save barely-trained weights that
    # claim N steps of training. Start over and say so.
    resumed = bool(resume_step) and (output_dir / "adapter_config.json").exists()
    if resume_step and not resumed:
        report(
            f"asked to resume {mix.module} at step {resume_step}, but there is no adapter "
            f"in {output_dir}; starting this module over"
        )
        resume_step = 0
    if resumed:
        model = PeftModel.from_pretrained(model, str(output_dir), is_trainable=True)
        report(f"resuming from step {resume_step}")
    else:
        model = get_peft_model(model, lora)
    model.train()
    outcome.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    outcome.detail["resumed"] = resumed

    micro = max(int(config.batch_size or plan.batch_size or 1), 1)
    accum = max(int(config.grad_accum or plan.grad_accum or 1), 1)
    per_epoch = max(len(encoded) // (micro * accum), 1)
    total_steps = min(int(per_epoch * max(config.epochs, 0.1)) or 1, config.max_steps)
    warmup = max(int(total_steps * config.warmup_ratio), 1) if total_steps > 4 else 0

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = _optimizer(trainable, config.learning_rate, plan.optimizer)

    report(
        f"training {mix.module}: {total_steps} steps, {outcome.examples} examples, "
        f"rank {rank}",
        steps=total_steps,
        examples=outcome.examples,
    )

    order = list(range(len(encoded)))
    cursor = (resume_step * micro * accum) % len(order)
    losses: list[float] = []
    step = resume_step
    deadline = started + config.max_seconds

    while step < total_steps:
        if stop():
            outcome.preempted = True
            outcome.stopped_by = "asked to stop"
            break
        if time.monotonic() > deadline:
            outcome.stopped_by = f"hit training.max_seconds ({config.max_seconds:.0f}s)"
            break

        lr = _lr_at(step, total_steps, base=config.learning_rate, warmup=warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            batch = []
            for _ in range(micro):
                batch.append(encoded[order[cursor % len(order)]])
                cursor += 1
            input_ids, labels, mask, sample_weights = _collate(
                batch, tokenizer.pad_token_id or 0, torch
            )
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            mask = mask.to(model.device)
            sample_weights = sample_weights.to(model.device)

            logits = model(input_ids=input_ids, attention_mask=mask).logits[:, :-1, :]
            target = labels[:, 1:]
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                target.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(target.shape)
            counted = (target != -100).float()
            per_sample = (per_token * counted).sum(1) / counted.sum(1).clamp(min=1.0)
            loss = (per_sample * sample_weights).mean() / accum
            loss.backward()
            step_loss += float(loss.detach()) * accum / accum

        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        optimizer.step()
        step += 1
        losses.append(step_loss)
        if outcome.loss_start is None:
            outcome.loss_start = step_loss
        outcome.loss_end = sum(losses[-5:]) / len(losses[-5:])

        report(
            f"step {step}/{total_steps}, loss {step_loss:.3f}",
            step=step,
            total=total_steps,
            loss=round(step_loss, 4),
            lr=lr,
        )
        if config.checkpoint_every_steps and step % config.checkpoint_every_steps == 0:
            _save(model, output_dir)
            report(f"checkpointed at step {step}", checkpoint_step=step)

    outcome.steps = step
    if step > resume_step or resumed:
        _save(model, output_dir)
        outcome.path = output_dir
    outcome.seconds = round(time.monotonic() - started, 1)
    outcome.detail.update(
        {
            "rank": rank,
            "total_steps_planned": total_steps,
            "micro_batch": micro,
            "grad_accum": accum,
            "learning_rate": config.learning_rate,
            "seq_len": seq_len,
            "losses": [round(v, 4) for v in losses[-20:]],
        }
    )
    if not outcome.stopped_by:
        outcome.stopped_by = "finished"

    del optimizer
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return outcome


def _optimizer(params: list[Any], lr: float, preferred: str):
    """8-bit Adam when bitsandbytes offers it: the states are half the VRAM."""
    if "8bit" in (preferred or "") or "paged" in (preferred or ""):
        try:
            import bitsandbytes as bnb

            if "paged" in preferred:
                return bnb.optim.PagedAdamW8bit(params, lr=lr)
            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception:  # noqa: BLE001 - fall back rather than fail the night
            pass
    import torch

    return torch.optim.AdamW(params, lr=lr)


def _save(model: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)


def write_meta(
    output_dir: Path,
    *,
    module: str,
    version: str,
    plan: Plan,
    outcome: TrainOutcome,
    mix: Mix,
    extra: dict[str, Any] | None = None,
) -> Path:
    """The adapter's own record of what it is. Read back by the loader and the gate."""
    meta = {
        "module": module,
        "version": version,
        "base_model": plan.base_model,
        "quantization": plan.quantization,
        "lora_rank": outcome.detail.get("rank", plan.lora_rank),
        "examples": outcome.examples,
        "fresh_examples": mix.fresh,
        "replayed_examples": mix.replay,
        "by_kind": mix.by_kind,
        "steps": outcome.steps,
        "loss_start": outcome.loss_start,
        "loss_end": outcome.loss_end,
        "seconds": outcome.seconds,
        "trainable_params": outcome.trainable_params,
        "stopped_by": outcome.stopped_by,
        **(extra or {}),
    }
    path = output_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path
