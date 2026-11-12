"""A deterministic in-memory backend.

Lets the harness, the daemon, the router and the adapter machinery be tested without a
GPU or any weights — and makes failure modes (a model that will not load, an adapter
that will not attach) easy to provoke on purpose.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

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


class FakeBackend(InferenceBackend):
    name = "fake"

    def __init__(
        self,
        *,
        base_model: str = "fake-model",
        reply: str | None = None,
        tokens_per_second: float = 42.0,
        fail_to_load: str | None = None,
        fail_adapter: str | None = None,
        load_delay: float = 0.0,
        embed_fails: str | None = None,
    ):
        self.base_model = base_model
        self.reply = reply
        self.tokens_per_second = tokens_per_second
        self.fail_to_load = fail_to_load
        self.fail_adapter = fail_adapter
        self.load_delay = load_delay
        self.embed_fails = embed_fails
        #: Whether this fake pretends its template understands tools, and what it was
        #: offered each time it rendered a prompt.
        self.tools_supported = True
        self.offered_tools: list[list[str]] = []

        self._loaded = False
        self._adapters: dict[str, str] = {}
        self._active: tuple[str, ...] = ()
        self.load_count = 0
        self.unload_count = 0
        self.prompts: list[list[Message]] = []
        self.params: list[GenerationParams] = []
        self.adapter_events: list[tuple[str, str]] = []
        self._last_generation: Generation | None = None

    # --- lifecycle ------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self.load_delay:
            await asyncio.sleep(self.load_delay)
        if self.fail_to_load:
            raise BackendError(self.fail_to_load)
        self._loaded = True
        self.load_count += 1

    async def unload(self) -> None:
        self._loaded = False
        self._adapters.clear()
        self._active = ()
        self.unload_count += 1

    # --- generation -----------------------------------------------------
    def _answer(self, messages: Sequence[Message]) -> str:
        if self.reply is not None:
            return self.reply
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        modules = ",".join(self._active) or "none"
        return f"fake answer to {last!r} using [{modules}]"

    async def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[Chunk]:
        if not self._loaded:
            raise BackendNotLoaded()
        self.prompts.append(list(messages))
        self.params.append(params)
        # The real backend renders the prompt (with any tools) before generating, and
        # tests assert on what the model was offered — so this fake does too.
        self.render_prompt(messages, tools=params.tools)

        words = self._answer(messages).split(" ")
        matcher = StopSequenceMatcher(stop=tuple(params.stop))
        finish = "stop"
        index = 0
        produced = 0
        for position, word in enumerate(words):
            if produced >= params.max_tokens:
                finish = "length"
                break
            piece = word + (" " if position < len(words) - 1 else "")
            result = matcher.feed(piece)
            if result.emit:
                yield Chunk(text=result.emit, index=index)
                index += 1
            produced += 1
            if result.stopped:
                finish = "stop_sequence"
                break
        else:
            tail = matcher.flush()
            if tail:
                yield Chunk(text=tail, index=index)
                index += 1

        self._last_generation = Generation(
            text=matcher.text,
            finish_reason=finish,
            prompt_tokens=sum(self.count_tokens(m["content"]) for m in messages),
            completion_tokens=produced,
            time_to_first_token_ms=1.0,
            tokens_per_second=self.tokens_per_second,
            adapters=dict(self.active_adapters),
            base_model=self.base_model,
            vram_peak_mb=1234.5,
            meta={"cached_prompt_tokens": 0},
        )
        yield Chunk(text="", index=index, finish_reason=finish)

    async def generate(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> Generation:
        async for _ in self.stream(messages, params):
            pass
        assert self._last_generation is not None
        return self._last_generation

    # --- tokens ---------------------------------------------------------
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """A hashed bag-of-words vector: deterministic, and similar text is similar.

        Not an embedding in any meaningful sense — but it has the one property the
        tests need, which is that two ways of saying the same words come out close
        together and unrelated text does not.
        """
        if self.embed_fails:
            raise NotImplementedError(self.embed_fails)
        import math
        import zlib

        dimensions = 128
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * dimensions
            for word in str(text).lower().split():
                # crc32, not hash(): Python's hash is randomised per process, which
                # would make these vectors — and any test that compares them — depend
                # on the interpreter's seed.
                vector[zlib.crc32(word.encode("utf-8")) % dimensions] += 1.0
            length = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / length for v in vector])
        return out

    def count_tokens(self, text: str) -> int:
        return max(len(text.split()), 1)

    @property
    def supports_tools(self) -> bool:
        return self.tools_supported

    def render_prompt(
        self, messages: Sequence[Message], *, tools: Sequence[dict[str, Any]] = ()
    ) -> str:
        self.offered_tools.append([t["function"]["name"] for t in tools])
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    # --- adapters -------------------------------------------------------
    async def load_adapter(self, module: str, path: Any, *, version: str) -> None:
        if not self._loaded:
            raise BackendNotLoaded()
        if self.fail_adapter == module:
            raise BackendError(f"adapter {module} is broken (on purpose)")
        if not Path(path).exists():
            raise BackendError(f"{path} does not exist")
        self._adapters[module] = version
        self.adapter_events.append(("load", f"{module}@{version}"))

    async def unload_adapter(self, module: str) -> None:
        self._adapters.pop(module, None)
        self._active = tuple(a for a in self._active if a != module)
        self.adapter_events.append(("unload", module))

    async def set_active_adapters(self, modules: Sequence[str]) -> None:
        self._active = tuple(m for m in modules if m in self._adapters)
        self.adapter_events.append(("activate", ",".join(self._active)))

    @property
    def active_adapters(self) -> dict[str, str]:
        return {m: self._adapters[m] for m in self._active}

    @property
    def loaded_adapters(self) -> dict[str, str]:
        return dict(self._adapters)

    # --- reporting ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "responder": self.name,
            "base_model": self.base_model,
            "quantization": "fake",
            "loaded": self._loaded,
            "adapters": self.active_adapters,
            "adapters_loaded": dict(self._adapters),
            "attention": "fake",
            "dtype": "fake",
            "notes": [],
        }


def write_fake_adapter(
    root: Path, module: str, version: str, *, base_model: str = "fake-model", **meta: Any
) -> Path:
    """Create a directory that looks enough like a PEFT adapter to be activated."""
    import json

    path = root / module / version
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": meta.get("r", 16)}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    (path / "meta.json").write_text(
        json.dumps({"module": module, "version": version, "base_model": base_model, **meta}),
        encoding="utf-8",
    )
    return path
