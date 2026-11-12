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

    # --- measuring ------------------------------------------------------
    def benchmark(self, *, max_tokens: int = 96, repeats: int = 1) -> dict[str, Any]:
        """Tokens per second on the fixed benchmark prompts.

        Deliberately *not* recorded into ``Measurements``: a self-optimisation trial
        may be running the system in a configuration it is about to throw away, and the
        planner's speed measurement must come from the configuration actually in use.
        """
        from gustav.inference.bench import run_benchmark

        if not self.available:
            raise RuntimeError(self.error or "no model to benchmark")
        result = self._run(
            run_benchmark(
                self._harness,
                max_tokens=max_tokens,
                repeats=repeats,
                include_long_prompt=False,
                measurements=None,
            )
        )
        return {
            "tokens_per_second": result.tokens_per_second,
            "time_to_first_token_ms": result.time_to_first_token_ms,
            "vram_peak_mb": result.vram_peak_mb,
            "samples": len(result.samples),
        }

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeddings for the gate's semantic comparison. Raises if unsupported."""
        if not self.available:
            raise RuntimeError(self.error or "no model to embed with")
        return self._run(self._harness.backend.embed(list(texts)))

    def quality(self, items: Sequence[tuple[str, str]]) -> float | None:
        """Share of (prompt, expected) pairs the model answers acceptably.

        The same blunt prose comparison the gate uses, on the same held-out material —
        the point is not to grade the model but to notice a setting that makes the
        answers worse.
        """
        from gustav.learn.bench import BenchItem, score_answer

        if not items or not self.available:
            return None
        answers = self.evaluate([prompt for prompt, _ in items], temperature=0.0,
                                max_tokens=128)
        passed = 0
        for (prompt, expected), answer in zip(items, answers, strict=True):
            item = BenchItem(prompt=prompt, expected=expected, verifier={"match": "prose"})
            ok, _why = score_answer(item, answer)
            passed += int(ok)
        return round(passed / len(items), 4)

    def ask_batch(
        self,
        prompts: Sequence[str | Sequence[Message]],
        *,
        system: str | None = None,
        **overrides: Any,
    ) -> list[str | None]:
        return [self.ask(p, system=system, **overrides) for p in prompts]
