"""The model, inside a sleep step's child process.

A sleep step cannot borrow the daemon's harness: it runs in another process, and the
whole point of the GPU lease is that the daemon has given the card up before the step
starts. So a step that needs the model builds its own backend, uses it, and drops it.

Three deliberate properties:

* **Loading is allowed to fail.** No weights, no GPU, a half-pulled checkpoint — every
  caller checks ``available`` and has a deterministic path for when it is False. A
  night without a model does less; it does not crash.
* **Nothing the model says is trusted.** ``ask`` returns text. Whether that text is
  worth keeping is decided by a verifier, a citation check or the user — never here.
* **It is synchronous.** Step bodies are plain functions; the async backend is driven
  on a private event loop so no step has to be written in async for no reason.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from gustav.inference.backend import GenerationParams, Message

if TYPE_CHECKING:  # pragma: no cover
    from gustav.sleep.stages._base import StepEnv


class StepModel:
    def __init__(self, env: "StepEnv"):
        self.env = env
        self._loop: asyncio.AbstractEventLoop | None = None
        self._harness: Any = None
        self._error: str | None = None
        self._tried = False
        self.calls = 0

    # --- lifecycle ------------------------------------------------------
    @property
    def error(self) -> str | None:
        self._ensure()
        return self._error

    @property
    def available(self) -> bool:
        self._ensure()
        return self._harness is not None

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        env = self.env
        if not env.config.sleep.use_model:
            self._error = "sleep.use_model is off"
            return
        if str(env.config.inference.get("responder", "auto")).lower() == "echo":
            self._error = "inference.responder is 'echo', so there is no model to ask"
            return
        try:
            self._harness = self._build()
        except Exception as exc:  # noqa: BLE001 - a step must survive this
            self._error = f"{type(exc).__name__}: {exc}"
            self._harness = None
            env.log(f"no model for {env.step}: {self._error}", level="warning")

    def _build(self) -> Any:
        from gustav.hw.service import HardwareService
        from gustav.inference.harness import Harness

        env = self.env
        hardware = HardwareService(env.config, env.paths, env.db, env.events)
        harness = Harness(
            config=env.config,
            paths=env.paths,
            hardware=hardware,
            events=env.events,
            db=env.db,
        )
        env.progress("loading the model")
        self._run(harness.load())
        return harness

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def close(self) -> None:
        if self._harness is not None:
            try:
                self._run(self._harness.unload())
            except Exception:  # noqa: BLE001 - we are on the way out
                pass
            self._harness = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    # --- asking ---------------------------------------------------------
    def params(self, **overrides: Any) -> GenerationParams:
        cfg = self.env.config
        base = GenerationParams(
            max_tokens=cfg.sleep.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
        )
        return base.merged(**overrides)

    def ask(
        self,
        prompt: str | Sequence[Message],
        *,
        system: str | None = None,
        **overrides: Any,
    ) -> str | None:
        """One answer, or None if there is no model or generation failed."""
        if not self.available:
            return None
        messages: list[Message]
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [dict(m) for m in prompt]
        if system and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system})
        try:
            generations = self._run(
                self._harness.evaluate([messages], params=self.params(**overrides))
            )
        except Exception as exc:  # noqa: BLE001
            self.env.log(f"generation failed: {type(exc).__name__}: {exc}", level="warning")
            return None
        self.calls += 1
        text = generations[0].text if generations else None
        return text.strip() if text else None

    def evaluate(
        self,
        prompts: Sequence[str | Sequence[Message]],
        *,
        candidate: tuple[str, str] | None = None,
        system: str | None = None,
        **overrides: Any,
    ) -> list[str | None]:
        """Answer every prompt, optionally with a candidate adapter version loaded.

        The harness restores whatever was active afterwards even if the candidate fails
        to load, which is the property the regression gate depends on: a measurement
        must not be able to leave the served model changed.
        """
        if not self.available:
            return [None] * len(prompts)
        messages: list[list[Message]] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                conversation: list[Message] = [{"role": "user", "content": prompt}]
            else:
                conversation = [dict(m) for m in prompt]
            if system and not any(m.get("role") == "system" for m in conversation):
                conversation.insert(0, {"role": "system", "content": system})
            messages.append(conversation)
        try:
            generations = self._run(
                self._harness.evaluate(
                    messages, params=self.params(**overrides), candidate=candidate
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.env.log(
                f"evaluation failed ({candidate or 'baseline'}): "
                f"{type(exc).__name__}: {exc}",
                level="warning",
            )
            return [None] * len(prompts)
        self.calls += len(messages)
        return [(g.text or "").strip() if g is not None else None for g in generations]

    def ask_batch(
        self,
        prompts: Sequence[str | Sequence[Message]],
        *,
        system: str | None = None,
        **overrides: Any,
    ) -> list[str | None]:
        return [self.ask(p, system=system, **overrides) for p in prompts]
