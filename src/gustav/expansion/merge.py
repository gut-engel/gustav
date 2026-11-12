"""Folding an adapter into the base checkpoint.

The one operation in the whole system that rewrites the weights everything else is
built on, which is why it is off by default, always needs approval, and always takes a
snapshot that *includes the base model*.

Two things about it that are easy to get wrong:

* **It cannot be done in the quantised weights the server runs.** A 4-bit NF4 linear
  layer has no fp16 matrix to add a LoRA delta to. So the merge loads the original
  checkpoint from disk in its own dtype, merges there, and writes a new checkpoint —
  which is also why it needs the memory for the unquantised model.
* **The architecture does not change.** Merging ``W + BA`` leaves every shape as it
  was, so the ladder's numbers, the hardware plan and the model id all stay valid. The
  merged checkpoint replaces the old one under the same id, atomically, and the
  adapters that were trained against the *old* base are archived rather than kept —
  they no longer describe a delta from these weights.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

#: Files that belong to a checkpoint but are not weights, copied across unchanged.
SIDECARS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)


class MergeError(RuntimeError):
    pass


def merge_adapter_into_base(
    *,
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    dtype: str = "bfloat16",
    device: str = "cpu",
    progress: Any = None,
) -> dict[str, Any]:
    """Write ``base_dir`` + ``adapter_dir`` into ``output_dir``. Returns what it did.

    ``device="cpu"`` is the default on purpose: this runs while the GPU may be serving,
    it is not latency-sensitive, and the unquantised model is the thing that does not
    fit on a small card beside everything else.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not (base_dir / "config.json").exists():
        raise MergeError(f"{base_dir} does not look like a checkpoint (no config.json)")
    if not any((adapter_dir / name).exists() for name in
               ("adapter_config.json", "adapter_model.safetensors")):
        raise MergeError(f"{adapter_dir} does not look like an adapter")

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)

    say(f"loading {base_dir.name} unquantised on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch_dtype,
        device_map={"": device},
    )
    say(f"applying {adapter_dir.name}")
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    say("merging")
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    say(f"writing the merged checkpoint to {output_dir}")
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    copied = []
    for name in SIDECARS:
        source = base_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
            copied.append(name)

    weights = sorted(p.name for p in output_dir.glob("*.safetensors"))
    if not weights:
        raise MergeError(f"the merge produced no weight files in {output_dir}")

    del merged, model
    import gc

    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "base": str(base_dir),
        "adapter": str(adapter_dir),
        "output": str(output_dir),
        "dtype": dtype,
        "weight_files": weights,
        "sidecars": copied,
        "size_mb": round(
            sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file()) / (1024 * 1024),
            1,
        ),
    }
