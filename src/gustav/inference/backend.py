"""The inference backend interface.

One abstract class, one real implementation (``TransformersBackend``), and a fake for
tests. Everything above this line — the harness, the server, the sleep pipeline's
evaluation hook — talks only to this interface, so a faster backend later (custom
kernels, a paged KV cache) is a new class rather than a rewrite.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

#: A chat turn, in the shape the model's own chat template expects.
Message = dict[str, str]


@dataclass
class GenerationParams:
    """Sampling settings for one request."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    #: Strings that end generation. Matched across token boundaries.
    stop: tuple[str, ...] = ()
    seed: int | None = None
    #: Adapter modules to apply, in order. Empty means "whatever the router picks".
    adapters: tuple[str, ...] = ()
    #: Tool definitions offered to the model, in the shape a chat template's ``tools=``
    #: argument expects. Empty means the model is not told about any tools at all —
    #: which is the default, and what "tools disabled" looks like from down here.
    tools: tuple[dict[str, Any], ...] = ()

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    def merged(self, **overrides: Any) -> "GenerationParams":
        data = {**self.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        if isinstance(data.get("stop"), (list, set)):
            data["stop"] = tuple(data["stop"])
        if isinstance(data.get("adapters"), (list, set)):
            data["adapters"] = tuple(data["adapters"])
        if isinstance(data.get("tools"), list):
            data["tools"] = tuple(data["tools"])
        return GenerationParams(**data)


@dataclass
class Chunk:
    """One streamed piece of a response."""

    text: str
    index: int
    token_id: int | None = None
    finish_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.finish_reason is not None


@dataclass
class Generation:
    """A finished response, with the numbers the telemetry and the gate want."""

    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    adapters: dict[str, str] = field(default_factory=dict)
    base_model: str = ""
    vram_peak_mb: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BackendError(RuntimeError):
    """Something went wrong inside the backend; the daemon stays up."""


class BackendNotLoaded(BackendError):
    def __init__(self) -> None:
        super().__init__("the model is not loaded (the GPU may be held by a sleep step)")


class InferenceBackend(abc.ABC):
    """What the rest of the system may assume about a model."""

    name: str = "backend"

    # --- lifecycle ------------------------------------------------------
    @property
    @abc.abstractmethod
    def loaded(self) -> bool: ...

    @abc.abstractmethod
    async def load(self) -> None:
        """Bring the model into VRAM. Must pass the memory pre-flight check first."""

    @abc.abstractmethod
    async def unload(self) -> None:
        """Free the GPU completely, so a training child can have it."""

    # --- generation -----------------------------------------------------
    @abc.abstractmethod
    def stream(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> AsyncIterator[Chunk]:
        """Yield chunks as they are produced. The one primitive; ``generate`` wraps it."""

    async def generate(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> Generation:
        pieces: list[str] = []
        finish = "stop"
        last: Chunk | None = None
        async for chunk in self.stream(messages, params):
            pieces.append(chunk.text)
            if chunk.finish_reason:
                finish = chunk.finish_reason
            last = chunk
        result = self._finish(("".join(pieces)), finish, last)
        return result

    def _finish(self, text: str, finish: str, last: Chunk | None) -> Generation:
        return Generation(text=text, finish_reason=finish, base_model=self.describe().get(
            "base_model", ""
        ))

    # --- tokens ---------------------------------------------------------
    @abc.abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abc.abstractmethod
    def render_prompt(
        self, messages: Sequence[Message], *, tools: Sequence[dict[str, Any]] = ()
    ) -> str:
        """Apply the model's own chat template. No hardcoded formats anywhere.

        ``tools`` is passed to the template, which is how a model that was trained to
        call tools learns what it may call. A template that does not accept tools gets
        the prompt without them, and the harness is told so it does not offer any.
        """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Mean-pooled hidden states, normalised. Optional: not every backend has them.

        The gate uses these to compare a free-text answer with its reference, which is
        the weakest link in the whole measurement. A backend that cannot produce them
        raises, and the gate falls back to word overlap and says so.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot produce embeddings")

    # --- adapters -------------------------------------------------------
    @abc.abstractmethod
    async def load_adapter(self, module: str, path: Any, *, version: str) -> None: ...

    @abc.abstractmethod
    async def unload_adapter(self, module: str) -> None: ...

    @abc.abstractmethod
    async def set_active_adapters(self, modules: Sequence[str]) -> None: ...

    @property
    @abc.abstractmethod
    def active_adapters(self) -> dict[str, str]:
        """module -> version, for the turn log and the before/after comparison."""

    # --- reporting ------------------------------------------------------
    @abc.abstractmethod
    def describe(self) -> dict[str, Any]: ...

    def health(self) -> tuple[bool, str]:
        if not self.loaded:
            return False, f"{self.name} is not loaded"
        return True, f"{self.name} is loaded"
