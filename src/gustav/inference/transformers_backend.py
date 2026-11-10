"""The real backend: HuggingFace transformers + PEFT, with our own generation loop.

No Ollama, no llama.cpp server, no vLLM. The base model is loaded quantised according
to the hardware plan, adapters are loaded straight from ``adapters/<module>/vNNN`` with
no conversion step, and tokens are produced by a loop we control — so sampling, stop
sequences and streaming behave exactly as specified rather than as a library's defaults.

Two deliberate choices worth knowing about:

* **Nothing is downloaded at load time.** Weights are loaded with
  ``local_files_only=True`` from ``models/<id>`` (or the local HF cache). Fetching a
  model is an explicit, separate act (``ai model pull``), so starting the daemon never
  reaches the network.
* **The pre-flight check runs first.** If the plan does not fit the card, loading is
  refused with an explanation instead of dying inside CUDA.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gustav.hw.ladder import ModelSpec, QuantSpec
from gustav.hw.memory import inference_estimate
from gustav.hw.planner import Plan
from gustav.hw.preflight import check_vram, vram_budget
from gustav.hw.types import Hardware
from gustav.inference.backend import (
    BackendError,
    BackendNotLoaded,
    Chunk,
    Generation,
    GenerationParams,
    InferenceBackend,
    Message,
)
from gustav.inference.stopping import StopSequenceMatcher

MIB = 1024 * 1024


@dataclass
class LoadReport:
    path: str
    load_seconds: float
    vram_after_mb: float
    attention: str
    dtype: str
    quantization: str
    local_only: bool
    notes: list[str]


class ModelNotAvailable(BackendError):
    """The weights are not on disk, and we do not download behind your back."""

    def __init__(self, model_id: str, repo: str, searched: list[Path]):
        locations = "\n  ".join(str(p) for p in searched)
        super().__init__(
            f"the weights for {model_id} ({repo}) are not on this machine.\n"
            f"Looked in:\n  {locations}\n"
            f"Fetch them once with:  ai model pull {model_id}"
        )
        self.model_id = model_id
        self.repo = repo


class ModelIncomplete(BackendError):
    """Half a checkpoint. Usually an interrupted or still-running download."""

    def __init__(self, path: Path, missing: list[str]):
        shown = ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
        super().__init__(
            f"the checkpoint at {path} is incomplete: {len(missing)} weight file(s) are "
            f"missing ({shown}).\n"
            f"Finish the download with:  ai model pull {path.name}"
        )
        self.path = path
        self.missing = missing


def missing_weight_files(path: Path) -> list[str]:
    """Which files a sharded checkpoint promises but does not have."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        import json

        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        except (json.JSONDecodeError, OSError):
            return ["model.safetensors.index.json (unreadable)"]
        shards = sorted(set(weight_map.values()))
        return [name for name in shards if not (path / name).exists()]
    if any(path.glob("*.safetensors")) or any(path.glob("*.bin")):
        return []
    return ["model.safetensors"]


class TransformersBackend(InferenceBackend):
    name = "transformers"

    def __init__(
        self,
        *,
        plan: Plan,
        model: ModelSpec,
        quant: QuantSpec,
        hardware: Hardware,
        models_dir: Path,
        adapters_dir: Path,
        reserve_fraction: float = 0.15,
        prefix_cache: bool = True,
        thinking: bool = False,
        logger=None,
    ):
        from gustav.core.logs import get_logger

        self.plan = plan
        self.spec = model
        self.quant = quant
        self.hardware = hardware
        self.models_dir = Path(models_dir)
        self.adapters_dir = Path(adapters_dir)
        self.reserve_fraction = reserve_fraction
        self.prefix_cache_enabled = prefix_cache
        #: Hybrid reasoning models (Qwen3 and friends) emit a <think> block unless the
        #: chat template is told otherwise. Off for chat; the sleep pipeline turns it on
        #: for practice tasks, where the reasoning is the point.
        self.thinking = thinking
        self._template_supports_thinking: bool | None = None
        self.log = logger or get_logger("inference")

        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._device_index: int = (plan.inference_gpus or [0])[0]
        self._adapters: dict[str, str] = {}
        self._active: tuple[str, ...] = ()
        self._load_report: LoadReport | None = None
        #: One generation at a time per model: the GPU is not shared within a process.
        self._gpu_lock = asyncio.Lock()
        self._prefix: _PrefixCache | None = None
        self._stacking_supported: bool | None = None
        #: Whether the model's forward() can slice logits to the last position. Without
        #: it, prefilling a 3000-token prompt materialises 3000 x vocab logits — about a
        #: gigabyte of VRAM for nothing, since only the last row is ever sampled.
        self._logits_kwarg: str | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def resolve_path(self) -> Path:
        """Where the weights are, or a clear refusal."""
        candidates = [
            self.models_dir / self.spec.id,
            self.models_dir / self.spec.repo.replace("/", "--"),
        ]
        for candidate in candidates:
            if (candidate / "config.json").exists():
                missing = missing_weight_files(candidate)
                if missing:
                    raise ModelIncomplete(candidate, missing)
                return candidate
        # The HF cache counts as local, so a model pulled by any tool works.
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(self.spec.repo, "config.json")
        if isinstance(cached, str) and Path(cached).exists():
            return Path(cached).parent
        raise ModelNotAvailable(self.spec.id, self.spec.repo, candidates)

    def preflight(self) -> None:
        """Refuse before CUDA has a chance to fail."""
        if not self.hardware.has_gpu:
            raise BackendError(
                "no GPU was detected; this backend needs one "
                "(set hardware.allow_training: false and use a smaller model for CPU use)"
            )
        index = self.plan.inference_gpus[0] if self.plan.inference_gpus else 0
        gpu = self.hardware.gpu(index)
        estimate = inference_estimate(
            self.spec,
            self.quant,
            seq_len=self.plan.max_seq_len,
            kv_dtype=self.plan.kv_cache_dtype,
            adapter_count=max(len(self._adapters), 1),
        )
        result = check_vram(
            estimate,
            vram_budget(gpu, reserve_fraction=self.reserve_fraction),
            suggestions=[
                f"max_seq_len {self.plan.max_seq_len // 2}",
                "a smaller model (`ai plan-hardware --balance 0.2`)",
            ],
        )
        for warning in result.warnings:
            self.log.warning("pre-flight: %s", warning)
        result.raise_if_blocked()

    async def load(self) -> None:
        if self.loaded:
            return
        self.preflight()
        await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self.resolve_path()
        started = time.monotonic()
        notes: list[str] = []

        index = self.plan.inference_gpus[0] if self.plan.inference_gpus else 0
        # Bind this worker thread to the card before touching any CUDA statistics.
        torch.cuda.set_device(index)
        self._device = torch.device(f"cuda:{index}")
        self._device_index = index
        torch.cuda.reset_peak_memory_stats(index)

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "device_map": {"": index},
        }
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            self.plan.compute_dtype, torch.float16
        )
        quant_config = self._quantization_config(dtype)
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        else:
            kwargs["dtype"] = dtype

        attention = self.plan.attention
        for attempt in (attention, "sdpa", "eager"):
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(path), attn_implementation=attempt, **kwargs
                )
                if attempt != attention:
                    notes.append(f"{attention} was unavailable; using {attempt}")
                attention = attempt
                break
            except (ImportError, ValueError, RuntimeError) as exc:
                if attempt == "eager":
                    raise BackendError(f"could not load {path}: {exc}") from exc
                self.log.warning("attn_implementation=%s failed (%s); falling back", attempt, exc)

        self._model.eval()
        self._model.generation_config.use_cache = True
        self._logits_kwarg = self._detect_logits_kwarg()
        if self._logits_kwarg is None:
            notes.append(
                "this model's forward() cannot slice logits, so long prompts will use "
                "roughly seq x vocab extra VRAM during prefill"
            )
        if self.prefix_cache_enabled:
            self._prefix = _PrefixCache(logger=self.log)

        vram = torch.cuda.memory_allocated(index) / MIB
        self._load_report = LoadReport(
            path=str(path),
            load_seconds=round(time.monotonic() - started, 2),
            vram_after_mb=round(vram, 1),
            attention=attention,
            dtype=str(dtype).removeprefix("torch."),
            quantization=self.quant.id,
            local_only=True,
            notes=notes,
        )
        self.log.info(
            "loaded %s from %s in %.1fs (%s, %s, %.0f MiB VRAM)",
            self.spec.label, path, self._load_report.load_seconds, self.quant.id,
            attention, vram,
        )

    def _detect_logits_kwarg(self) -> str | None:
        import inspect

        try:
            parameters = inspect.signature(type(self._model).forward).parameters
        except (TypeError, ValueError):
            return None
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in parameters:
                return name
        return None

    def _forward_kwargs(self, *, prefill: bool) -> dict[str, Any]:
        """Only the last position's logits are ever sampled, so only ask for those."""
        if self._logits_kwarg is None:
            return {}
        return {self._logits_kwarg: 1}

    def _quantization_config(self, dtype: Any):
        if not self.quant.quantized:
            return None
        from transformers import BitsAndBytesConfig

        if self.quant.bytes_per_param <= 0.6:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    async def unload(self) -> None:
        """Free the GPU completely. A training child needs every megabyte."""
        if not self.loaded:
            return
        async with self._gpu_lock:
            await asyncio.to_thread(self._unload_blocking)

    def _unload_blocking(self) -> None:
        import torch

        self._prefix = None
        self._model = None
        self._tokenizer = None
        self._adapters.clear()
        self._active = ()
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.log.info("unloaded the model and released the GPU")

    # ------------------------------------------------------------------
    # prompting
    # ------------------------------------------------------------------
    def render_prompt(
        self, messages: Sequence[Message], *, thinking: bool | None = None
    ) -> str:
        """The model's own chat template — never a hardcoded format."""
        if self._tokenizer is None:
            raise BackendNotLoaded()
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        want_thinking = self.thinking if thinking is None else thinking
        if self._supports_thinking_flag():
            kwargs["enable_thinking"] = want_thinking
        try:
            return self._tokenizer.apply_chat_template(list(messages), **kwargs)
        except TypeError:
            self._template_supports_thinking = False
            return self._tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:  # noqa: BLE001 - a model without a template at all
            self.log.warning("no usable chat template (%s); falling back to a plain format", exc)
            lines = [f"{m['role']}: {m['content']}" for m in messages]
            return "\n".join([*lines, "assistant:"])

    def _supports_thinking_flag(self) -> bool:
        """Does this model's template take ``enable_thinking``? Asked once, of the template."""
        if self._template_supports_thinking is None:
            template = getattr(self._tokenizer, "chat_template", None) or ""
            if not isinstance(template, str):
                template = str(template)
            self._template_supports_thinking = "enable_thinking" in template
        return self._template_supports_thinking

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            # A rough stand-in so the context budget works before the model is loaded.
            return max(len(text) // 4, 1)
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    async def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[Chunk]:
        if not self.loaded:
            raise BackendNotLoaded()
        queue: asyncio.Queue[Chunk | BaseException | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(item: Chunk | BaseException | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        async with self._gpu_lock:
            worker = asyncio.create_task(
                asyncio.to_thread(self._generate_blocking, list(messages), params, emit)
            )
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                with contextlib.suppress(Exception):
                    await worker

    def _generate_blocking(
        self,
        messages: list[Message],
        params: GenerationParams,
        emit,
    ) -> None:
        """Our generation loop. Runs in a worker thread; streams through ``emit``."""
        import torch

        try:
            prompt = self.render_prompt(messages)
            input_ids = self._tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self._device)
            prompt_length = int(input_ids.shape[-1])

            generator = None
            if params.seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(params.seed)

            eos_ids = self._eos_ids()
            matcher = StopSequenceMatcher(stop=tuple(params.stop))
            produced: list[int] = []
            index = 0
            started = time.monotonic()
            first_token_at: float | None = None
            finish_reason = "length"
            decoded_so_far = ""

            with torch.inference_mode():
                past, cached_tokens = self._prime(input_ids)
                logits = past.pop("logits")
                cache = past.pop("past_key_values")
                next_input = None

                while len(produced) < params.max_tokens:
                    if next_input is not None:
                        outputs = self._model(
                            input_ids=next_input,
                            past_key_values=cache,
                            use_cache=True,
                            **self._forward_kwargs(prefill=False),
                        )
                        logits = outputs.logits
                        cache = outputs.past_key_values

                    from gustav.inference.sampling import sample_next

                    token = sample_next(
                        logits[:, -1, :].float(),
                        params,
                        generated=produced,
                        generator=generator,
                    )
                    token_id = int(token.item())
                    if token_id in eos_ids:
                        finish_reason = "stop"
                        break

                    produced.append(token_id)
                    if first_token_at is None:
                        first_token_at = time.monotonic()

                    text = self._tokenizer.decode(produced, skip_special_tokens=True)
                    piece = text[len(decoded_so_far) :]
                    decoded_so_far = text
                    result = matcher.feed(piece)
                    if result.emit:
                        emit(Chunk(text=result.emit, index=index))
                        index += 1
                    if result.stopped:
                        finish_reason = "stop_sequence"
                        break
                    next_input = token.view(1, 1)

                else:
                    finish_reason = "length"

            if self._prefix is not None:
                # The cache now holds prompt + completion, so record that: the next turn
                # re-sends this answer as history and can reuse all of it.
                self._prefix.extend(produced)

            tail = matcher.flush() if finish_reason != "stop_sequence" else ""
            if tail:
                emit(Chunk(text=tail, index=index))
                index += 1

            elapsed = max(time.monotonic() - (first_token_at or started), 1e-6)
            generation = Generation(
                text=matcher.text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_length,
                completion_tokens=len(produced),
                time_to_first_token_ms=round(
                    ((first_token_at or started) - started) * 1000, 2
                ),
                tokens_per_second=round(len(produced) / elapsed, 2) if produced else 0.0,
                adapters=dict(self.active_adapters),
                base_model=self.spec.id,
                vram_peak_mb=self._vram_peak_mb(),
                meta={"cached_prompt_tokens": cached_tokens},
            )
            emit(
                Chunk(
                    text="",
                    index=index,
                    finish_reason=finish_reason,
                )
            )
            self._last_generation = generation
        except BaseException as exc:  # noqa: BLE001 - hand it to the awaiting coroutine
            emit(exc)
        finally:
            emit(None)

    def _prime(self, input_ids: Any) -> tuple[dict[str, Any], int]:
        """Run the prompt through the model, reusing a cached prefix when possible."""
        reuse, cache = (0, None)
        if self._prefix is not None:
            reuse, cache = self._prefix.lookup(input_ids)
        kwargs = self._forward_kwargs(prefill=True)
        if cache is not None and reuse > 0:
            remainder = input_ids[:, reuse:]
            outputs = self._model(
                input_ids=remainder, past_key_values=cache, use_cache=True, **kwargs
            )
        else:
            outputs = self._model(input_ids=input_ids, use_cache=True, **kwargs)
            reuse = 0
        if self._prefix is not None:
            self._prefix.store(input_ids, outputs.past_key_values)
        return (
            {"logits": outputs.logits, "past_key_values": outputs.past_key_values},
            reuse,
        )

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        config = getattr(self._model, "generation_config", None)
        for source in (
            getattr(config, "eos_token_id", None),
            getattr(self._tokenizer, "eos_token_id", None),
        ):
            if isinstance(source, int):
                ids.add(source)
            elif isinstance(source, (list, tuple)):
                ids.update(int(i) for i in source)
        return ids

    def _vram_peak_mb(self) -> float | None:
        try:
            import torch

            return round(torch.cuda.max_memory_allocated(self._device_index) / MIB, 1)
        except Exception:  # noqa: BLE001
            return None

    async def generate(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> Generation:
        self._last_generation = None
        async for _ in self.stream(messages, params):
            pass
        generation = getattr(self, "_last_generation", None)
        if generation is None:
            raise BackendError("generation produced no result")
        return generation

    # ------------------------------------------------------------------
    # adapters
    # ------------------------------------------------------------------
    async def load_adapter(self, module: str, path: Any, *, version: str) -> None:
        if not self.loaded:
            raise BackendNotLoaded()
        await asyncio.to_thread(self._load_adapter_blocking, module, Path(path), version)

    def _load_adapter_blocking(self, module: str, path: Path, version: str) -> None:
        from peft import PeftModel

        if not (path / "adapter_config.json").exists():
            raise BackendError(f"{path} does not look like a PEFT adapter")
        if isinstance(self._model, PeftModel) or hasattr(self._model, "load_adapter"):
            try:
                self._model.load_adapter(str(path), adapter_name=module)
            except Exception as exc:  # noqa: BLE001
                raise BackendError(f"could not load adapter {module} from {path}: {exc}") from exc
        else:
            self._model = PeftModel.from_pretrained(
                self._model, str(path), adapter_name=module, is_trainable=False
            )
        self._adapters[module] = version
        self.log.info("loaded adapter %s@%s", module, version)

    async def unload_adapter(self, module: str) -> None:
        if not self.loaded or module not in self._adapters:
            return
        await asyncio.to_thread(self._unload_adapter_blocking, module)

    def _unload_adapter_blocking(self, module: str) -> None:
        delete = getattr(self._model, "delete_adapter", None)
        if callable(delete):
            with contextlib.suppress(Exception):
                delete(module)
        self._adapters.pop(module, None)
        self._active = tuple(a for a in self._active if a != module)

    async def set_active_adapters(self, modules: Sequence[str]) -> None:
        if not self.loaded:
            raise BackendNotLoaded()
        wanted = tuple(m for m in modules if m in self._adapters)
        if wanted == self._active:
            return
        await asyncio.to_thread(self._set_active_blocking, wanted)

    def _set_active_blocking(self, modules: tuple[str, ...]) -> None:
        setter = getattr(self._model, "set_adapter", None)
        if not callable(setter):
            if modules:
                self.log.warning("this model cannot switch adapters; ignoring %s", modules)
            return
        if not modules:
            disable = getattr(self._model, "disable_adapter", None)
            if callable(disable):
                with contextlib.suppress(Exception):
                    self._model.disable_adapters()
            self._active = ()
            return
        if len(modules) == 1:
            setter(modules[0])
        else:
            # Stacking several adapters is a PEFT feature, not a given. Fall back to the
            # first one rather than silently applying something different.
            try:
                setter(list(modules))
                self._stacking_supported = True
            except (TypeError, ValueError) as exc:
                if self._stacking_supported is not False:
                    self.log.warning(
                        "this PEFT version cannot stack adapters (%s); using %s only",
                        exc, modules[0],
                    )
                self._stacking_supported = False
                setter(modules[0])
                modules = (modules[0],)
        self._active = modules

    @property
    def active_adapters(self) -> dict[str, str]:
        return {name: self._adapters[name] for name in self._active if name in self._adapters}

    @property
    def loaded_adapters(self) -> dict[str, str]:
        return dict(self._adapters)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        report = self._load_report
        return {
            "responder": self.name,
            "base_model": self.spec.id,
            "repo": self.spec.repo,
            "quantization": self.quant.id,
            "max_seq_len": self.plan.max_seq_len,
            "loaded": self.loaded,
            "adapters": self.active_adapters,
            "adapters_loaded": self.loaded_adapters,
            "attention": report.attention if report else self.plan.attention,
            "dtype": report.dtype if report else self.plan.compute_dtype,
            "path": report.path if report else None,
            "load_seconds": report.load_seconds if report else None,
            "vram_mb": report.vram_after_mb if report else None,
            "notes": report.notes if report else [],
        }

    def health(self) -> tuple[bool, str]:
        if not self.loaded:
            return False, "the model is not loaded"
        return True, f"{self.spec.label} loaded with {len(self._adapters)} adapter(s)"


class _PrefixCache:
    """Reuses the KV cache for the longest common prefix of consecutive prompts.

    A multi-turn chat re-sends the whole conversation every time, so the common prefix
    is nearly the entire prompt — which is exactly the expensive part. One entry is
    enough to capture that, and because the GPU is serialised by a lock there is no
    sharing problem.
    """

    def __init__(self, *, logger=None):
        self.ids: list[int] = []
        self.cache: Any = None
        self.log = logger
        self.hits = 0
        self.misses = 0

    def lookup(self, input_ids: Any) -> tuple[int, Any]:
        wanted = input_ids[0].tolist()
        if self.cache is None or not self.ids:
            self.misses += 1
            return 0, None
        shared = 0
        for mine, theirs in zip(self.ids, wanted, strict=False):
            if mine != theirs:
                break
            shared += 1
        # Reusing everything would leave nothing to feed the model, so keep one token.
        shared = min(shared, len(wanted) - 1, len(self.ids))
        if shared <= 0:
            self.misses += 1
            return 0, None
        cropped = self._crop(shared)
        if cropped is None:
            self.misses += 1
            return 0, None
        self.hits += 1
        return shared, cropped

    def _crop(self, length: int) -> Any:
        """Trim the cache down to ``length`` tokens.

        transformers 5's ``DynamicCache.crop`` takes the number of tokens to *remove* as
        a negative integer; older versions took an absolute length. Both are tried, so
        this keeps working across versions rather than silently disabling itself.
        """
        crop = getattr(self.cache, "crop", None)
        if not callable(crop):
            return None
        current = self._length()
        if current is None:
            return None
        if current == length:
            return self.cache
        try:
            crop(length - current)  # negative: remove this many from the end
        except (ValueError, TypeError):
            try:
                crop(length)  # older absolute-length API
            except Exception as exc:  # noqa: BLE001
                return self._give_up(exc)
        except Exception as exc:  # noqa: BLE001 - never let the cache break generation
            return self._give_up(exc)
        if self._length() != length:
            return self._give_up(
                RuntimeError(f"crop left {self._length()} tokens, wanted {length}")
            )
        return self.cache

    def _length(self) -> int | None:
        getter = getattr(self.cache, "get_seq_length", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:  # noqa: BLE001
                return None
        return len(self.ids) or None

    def _give_up(self, exc: BaseException) -> None:
        if self.log:
            self.log.debug("prefix cache unusable (%s); starting fresh", exc)
        self.cache = None
        self.ids = []
        return None

    def store(self, input_ids: Any, cache: Any) -> None:
        self.ids = input_ids[0].tolist()
        self.cache = cache

    def extend(self, token_ids: Sequence[int]) -> None:
        """Record tokens the model generated into the same cache."""
        self.ids.extend(int(t) for t in token_ids)

    def invalidate(self) -> None:
        self.ids = []
        self.cache = None

    def describe(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "prefix_hits": self.hits,
            "prefix_misses": self.misses,
            "prefix_hit_rate": round(self.hits / total, 3) if total else None,
            "cached_tokens": len(self.ids),
        }
