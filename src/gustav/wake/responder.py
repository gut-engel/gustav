"""The wake-phase responder interface.

Stage 1 answers with an echo so the daemon, scheduler and preemption logic can be
exercised end to end without a GPU. Stage 3 drops in the real inference harness
behind this same interface — including ``unload``, which is how the GPU lease is
honoured when training needs the card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Reply:
    text: str
    adapter_versions: dict[str, str] = field(default_factory=dict)
    base_model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Responder(Protocol):
    name: str

    @property
    def loaded(self) -> bool: ...

    async def load(self) -> None: ...

    async def unload(self) -> None:
        """Free the GPU. Called when a sleep step needs the card."""

    async def respond(
        self, *, session_id: int, message: str, history: list[dict[str, str]]
    ) -> Reply: ...

    def health(self) -> tuple[bool, str]: ...

    def describe(self) -> dict[str, Any]: ...


class EchoResponder:
    """Placeholder responder: no model, no GPU, deterministic answers."""

    name = "echo"

    def __init__(self) -> None:
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        self._loaded = True
        self.load_count += 1

    async def unload(self) -> None:
        self._loaded = False
        self.unload_count += 1

    async def respond(
        self, *, session_id: int, message: str, history: list[dict[str, str]]
    ) -> Reply:
        if not self._loaded:
            await self.load()
        return Reply(
            text=f"(echo, stage 1 placeholder) {message}",
            adapter_versions={},
            base_model="none",
            prompt_tokens=len(message.split()),
            completion_tokens=len(message.split()) + 5,
            tokens_per_second=None,
            meta={"turns_in_history": len(history)},
        )

    def health(self) -> tuple[bool, str]:
        return True, "echo responder is always healthy"

    def describe(self) -> dict[str, Any]:
        return {
            "responder": self.name,
            "base_model": None,
            "adapters": {},
            "loaded": self._loaded,
            "note": "stage 1 placeholder; the real harness arrives in stage 3",
        }
