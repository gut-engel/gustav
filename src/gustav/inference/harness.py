"""The harness: the thing the daemon actually talks to.

It owns the backend, the adapter store, the router, the tiered cache and the context
budget, and it turns "here is a message and some history" into "here are tokens" —
recording which weights produced them, so behaviour before and after a sleep can be
compared.

It also exposes the evaluation hook the sleep pipeline needs: ``evaluate()`` runs a set
of prompts against a *candidate* adapter and puts the active one back afterwards,
whatever happens.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gustav.config.schema import Config
from gustav.core.events import Events, Telemetry
from gustav.core.logs import get_logger
from gustav.hw.ladder import Ladder, ModelSpec, QuantSpec
from gustav.hw.planner import Plan
from gustav.hw.service import HardwareService
from gustav.inference.adapters import AdapterError, AdapterStore, AdapterVersion
from gustav.inference.backend import (
    BackendError,
    Chunk,
    Generation,
    GenerationParams,
    InferenceBackend,
    Message,
)
from gustav.inference.conversation import ContextBudget, Conversation
from gustav.inference.registry import Registry, load_registry
from gustav.inference.router import RouteDecision, Router, RouterLog
from gustav.inference.tiered_cache import Tier, TieredCache
from gustav.paths import Paths
from gustav.wake.responder import Reply


@dataclass
class HarnessStatus:
    loaded: bool
    base_model: str | None
    adapters: dict[str, str]
    plan: str | None
    notes: list[str] = field(default_factory=list)


class Harness:
    """Implements the ``Responder`` protocol with a real model behind it."""

    name = "harness"

    def __init__(
        self,
        *,
        config: Config,
        paths: Paths,
        hardware: HardwareService,
        events: Events,
        telemetry: Telemetry | None = None,
        db=None,
        backend: InferenceBackend | None = None,
        logger=None,
    ):
        self.config = config
        self.paths = paths
        self.hw = hardware
        self.events = events
        self.telemetry = telemetry
        self.db = db
        self.log = logger or get_logger("harness")

        self.adapters = AdapterStore(paths.adapters)
        self.registry: Registry = load_registry(paths.registry)
        self.router = Router(
            self.registry,
            threshold=config.tunable.router_threshold,
            available=set(self.adapters.modules()) or None,
        )
        self.router_log = RouterLog(db)
        self.cache: TieredCache | None = None
        self._backend: InferenceBackend | None = backend
        self._explicit_backend = backend is not None
        self._plan: Plan | None = None
        self._spec: ModelSpec | None = None
        self._quant: QuantSpec | None = None
        self._load_lock = asyncio.Lock()
        self._notes: list[str] = []
        self._last_route: RouteDecision | None = None
        #: Set when a load was attempted and failed. "Not loaded yet" is not a failure:
        #: the daemon's startup health check runs *before* the first load, and treating
        #: that as unhealthy would send it straight to safe mode.
        self._load_failed: str | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _ladder(self) -> Ladder:
        return self.hw._ladder_with_local_configs()

    def _build_backend(self) -> InferenceBackend:
        from gustav.inference.transformers_backend import TransformersBackend

        plan = self.hw.stored_plan()
        if plan is None:
            raise BackendError(
                "there is no hardware plan yet; run 'ai plan-hardware --apply' "
                "or start the daemon so it can plan"
            )
        ladder = self._ladder()
        spec = ladder.model(plan.base_model)
        quant = ladder.quant(plan.quantization)
        self._plan, self._spec, self._quant = plan, spec, quant
        return TransformersBackend(
            plan=plan,
            model=spec,
            quant=quant,
            hardware=self.hw.hardware,
            models_dir=self.paths.models,
            adapters_dir=self.paths.adapters,
            reserve_fraction=self.config.hardware.vram_reserve_fraction,
            prefix_cache=self.config.tunable.prefix_cache,
            thinking=False,
            logger=self.log,
        )

    @property
    def backend(self) -> InferenceBackend:
        if self._backend is None:
            self._backend = self._build_backend()
        return self._backend

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._backend is not None and self._backend.loaded

    async def load(self) -> None:
        async with self._load_lock:
            if self.loaded:
                return
            self._notes = []
            try:
                await self.backend.load()
            except Exception as exc:
                self._load_failed = f"{type(exc).__name__}: {exc}"
                raise
            self._load_failed = None
            self._init_cache()
            await self._load_active_adapters()
            self.events.emit(
                "inference.loaded",
                f"{self.backend.describe().get('base_model')} ready",
                **{
                    k: v
                    for k, v in self.backend.describe().items()
                    if k in ("quantization", "attention", "dtype", "vram_mb", "load_seconds")
                },
            )

    async def unload(self) -> None:
        async with self._load_lock:
            if self._backend is None:
                return
            await self._backend.unload()
            if not self._explicit_backend:
                # Drop the object too, so the next load re-reads the plan.
                self._backend = None
            self.events.emit("inference.unloaded", "the model released the GPU")

    def _init_cache(self) -> None:
        plan = self._plan or self.hw.stored_plan()
        if plan is None:
            return
        budget = plan.expansion
        self.cache = TieredCache(
            vram_budget_mb=budget.max_adapter_vram_mb or 512,
            ram_budget_mb=budget.warm_cache_mb,
        )
        for module in self.adapters.modules():
            version = self.adapters.active(module)
            if version is None:
                continue
            size = float(version.meta.get("size_mb") or 0.0) or self._estimate_adapter_mb()
            self.cache.register(module, size, tier=Tier.DISK, version=version.version)

    def _estimate_adapter_mb(self) -> float:
        if self._spec is None or self._plan is None:
            return 64.0
        params = self._spec.lora_params(self._plan.lora_rank)
        return round(params * 2 / (1024 * 1024), 2)

    async def _load_active_adapters(self) -> None:
        """Bring every module's active version into the model, within the budget."""
        for module in self.adapters.modules():
            version = self.adapters.active(module)
            if version is None or not version.complete:
                continue
            if not self._adapter_matches_base(version):
                continue
            try:
                await self.backend.load_adapter(module, version.path, version=version.version)
                if self.cache is not None:
                    self.cache.admit(module, target=Tier.VRAM)
            except BackendError as exc:
                self._notes.append(f"adapter {module}@{version.version} did not load: {exc}")
                self.log.error("adapter %s@%s failed to load: %s", module, version.version, exc)
        self.router.available = set(self.adapters.modules()) or None

    def _adapter_matches_base(self, version: AdapterVersion) -> bool:
        base = version.base_model
        # Ask the backend what it actually loaded rather than what we planned to load.
        current = self.backend.describe().get("base_model") or (
            self._spec.id if self._spec else None
        )
        if base and current and base != current:
            message = (
                f"adapter {version.module}@{version.version} was trained on {base}, "
                f"but the base model is {current}; skipping it"
            )
            self._notes.append(message)
            self.events.emit("inference.adapter_mismatch", message, level="warning")
            return False
        return True

    # ------------------------------------------------------------------
    # answering
    # ------------------------------------------------------------------
    def _params(self, **overrides: Any) -> GenerationParams:
        settings = self.config.inference or {}
        base = GenerationParams(
            max_tokens=int(settings.get("max_tokens", 512)),
            temperature=float(settings.get("temperature", 0.7)),
            top_p=float(settings.get("top_p", 0.95)),
            top_k=int(settings.get("top_k", 20)),
            min_p=float(settings.get("min_p", 0.0)),
            repetition_penalty=float(settings.get("repetition_penalty", 1.05)),
            stop=tuple(settings.get("stop", ())),
        )
        return base.merged(**overrides)

    def _budget(self) -> ContextBudget:
        plan = self._plan or self.hw.stored_plan()
        max_seq = plan.max_seq_len if plan else 2048
        return ContextBudget(
            max_seq_len=max_seq,
            reserve_for_completion=self._params().max_tokens,
        )

    async def _prepare(
        self, message: str, history: Sequence[Message], params: GenerationParams
    ) -> tuple[list[Message], RouteDecision]:
        conversation = Conversation.from_history(
            history, max_turns=self.config.wake.max_history_turns
        )
        conversation.add_user(message)
        decision = self.router.route(message, forced=params.adapters)
        self._last_route = decision
        await self._apply_route(decision)
        fitted = self._budget().fit(conversation.messages(), self.backend.count_tokens)
        for note in fitted.notes:
            self.log.debug("context: %s", note)
        return fitted.messages, decision

    async def _apply_route(self, decision: RouteDecision) -> None:
        wanted = tuple(
            m for m in decision.modules if m in getattr(self.backend, "loaded_adapters", {})
        )
        if self.cache is not None and wanted:
            for admission in self.cache.require(wanted, target=Tier.VRAM):
                if not admission.admitted:
                    self.log.warning("module %s stayed cold: %s", admission.name, admission.reason)
            wanted = tuple(m for m in wanted if self.cache.tier_of(m) is Tier.VRAM)
        with contextlib.suppress(BackendError):
            await self.backend.set_active_adapters(wanted)

    async def respond(
        self,
        *,
        session_id: int,
        message: str,
        history: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> Reply:
        params = params or self._params()
        messages, decision = await self._prepare(message, history, params)
        generation = await self.backend.generate(messages, params)
        self._record(generation, decision)
        return self._reply(generation, decision)

    async def stream(
        self,
        *,
        session_id: int,
        message: str,
        history: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> AsyncIterator[Chunk]:
        params = params or self._params()
        messages, decision = await self._prepare(message, history, params)
        async for chunk in self.backend.stream(messages, params):
            yield chunk
        generation = getattr(self.backend, "_last_generation", None)
        if generation is not None:
            self._record(generation, decision)

    def last_generation(self) -> Generation | None:
        return getattr(self.backend, "_last_generation", None)

    def _reply(self, generation: Generation, decision: RouteDecision) -> Reply:
        return Reply(
            text=generation.text,
            adapter_versions=generation.adapters,
            base_model=generation.base_model,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            tokens_per_second=generation.tokens_per_second,
            meta={
                "finish_reason": generation.finish_reason,
                "time_to_first_token_ms": generation.time_to_first_token_ms,
                "modules": list(decision.modules),
                "router_confidence": decision.confidence,
                "router_uncovered": decision.uncovered,
                "cached_prompt_tokens": generation.meta.get("cached_prompt_tokens", 0),
                "vram_peak_mb": generation.vram_peak_mb,
            },
        )

    def _record(self, generation: Generation, decision: RouteDecision) -> None:
        if self.telemetry is not None:
            self.telemetry.record(
                "inference.tokens_per_second",
                generation.tokens_per_second,
                model=generation.base_model,
                adapters=list(generation.adapters),
            )
            self.telemetry.record(
                "inference.time_to_first_token_ms", generation.time_to_first_token_ms
            )
            if generation.vram_peak_mb:
                self.telemetry.record("inference.vram_peak_mb", generation.vram_peak_mb)
        self.router_log.record(decision)
        if decision.uncovered:
            # A topic no module covers is one of the self-expansion triggers (section 8).
            self.events.emit(
                "router.uncovered",
                f"no module covers this request well (best score {decision.confidence:.2f})",
                level="debug",
                scores={k: round(v, 3) for k, v in decision.scores.items()},
            )

    # ------------------------------------------------------------------
    # the evaluation hook (used by the sleep gate, stage 5)
    # ------------------------------------------------------------------
    async def evaluate(
        self,
        prompts: Sequence[Sequence[Message]],
        *,
        params: GenerationParams | None = None,
        candidate: tuple[str, str] | None = None,
    ) -> list[Generation]:
        """Run prompts, optionally against a candidate adapter version.

        The active adapter is restored afterwards even if the candidate explodes, which
        is the property the regression gate depends on.
        """
        params = params or self._params(temperature=0.0, max_tokens=256)
        restore: tuple[str, str] | None = None
        previous_active = tuple(self.backend.active_adapters)

        if candidate is not None:
            module, version = candidate
            current = self.adapters.active(module)
            restore = (module, current.version) if current else None
            target = self.adapters.get(module, version)
            await self.backend.load_adapter(module, target.path, version=version)
            await self.backend.set_active_adapters([module])

        try:
            return [await self.backend.generate(list(prompt), params) for prompt in prompts]
        finally:
            if candidate is not None:
                module, _ = candidate
                with contextlib.suppress(Exception):
                    await self.backend.unload_adapter(module)
                if restore is not None:
                    with contextlib.suppress(Exception):
                        target = self.adapters.get(*restore)
                        await self.backend.load_adapter(
                            restore[0], target.path, version=restore[1]
                        )
                with contextlib.suppress(Exception):
                    await self.backend.set_active_adapters(previous_active)

    # ------------------------------------------------------------------
    # adapter control
    # ------------------------------------------------------------------
    async def activate_adapter(self, module: str, version: str) -> AdapterVersion:
        """Hot-swap a version in without restarting."""
        target = self.adapters.activate(module, version)
        if self.loaded:
            with contextlib.suppress(BackendError):
                await self.backend.unload_adapter(module)
            await self.backend.load_adapter(module, target.path, version=version)
            if self.cache is not None:
                self.cache.register(module, self._estimate_adapter_mb(), version=version)
                self.cache.admit(module, target=Tier.VRAM)
        self.events.emit(
            "adapter.activated", f"{module} is now {version}", module=module, version=version
        )
        return target

    async def rollback_adapter(self, module: str) -> AdapterVersion:
        target = self.adapters.rollback(module)
        if self.loaded:
            with contextlib.suppress(BackendError):
                await self.backend.unload_adapter(module)
            await self.backend.load_adapter(module, target.path, version=target.version)
        self.events.emit(
            "adapter.rolled_back",
            f"{module} rolled back to {target.version}",
            level="warning",
            module=module,
            version=target.version,
        )
        return target

    def adapter_status(self) -> dict[str, Any]:
        status = self.adapters.describe()
        loaded = getattr(self.backend, "loaded_adapters", {}) if self._backend else {}
        for module, info in status.items():
            info["loaded"] = module in loaded
            info["tier"] = str(self.cache.tier_of(module)) if self.cache else None
        return {
            "modules": status,
            "active_in_model": self.backend.active_adapters if self._backend else {},
            "cache": self.cache.describe() if self.cache else None,
            "registry": self.registry.names,
        }

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def health(self) -> tuple[bool, str]:
        if self._load_failed:
            return False, f"the model failed to load: {self._load_failed}"
        if self._backend is None or not self._backend.loaded:
            return True, "no model loaded yet (nothing has failed)"
        return self._backend.health()

    def describe(self) -> dict[str, Any]:
        if self._backend is None:
            plan = self.hw.stored_plan()
            return {
                "responder": self.name,
                "base_model": plan.base_model if plan else None,
                "adapters": {},
                "loaded": False,
                "plan": plan.summary() if plan else None,
                "notes": ["the model has not been loaded yet"],
            }
        described = dict(self._backend.describe())
        described["responder"] = self.name
        notes = [*described.get("notes", []), *self._notes]
        if not self._backend.loaded:
            notes.insert(0, "the model has not been loaded yet")
        if self._load_failed:
            notes.insert(0, f"the last load attempt failed: {self._load_failed}")
        described["notes"] = notes
        described["modules"] = self.registry.names
        if self._last_route is not None:
            described["last_route"] = list(self._last_route.modules)
        if self.cache is not None:
            described["adapter_cache"] = self.cache.describe()
        prefix = getattr(self._backend, "_prefix", None)
        if prefix is not None:
            described.update(prefix.describe())
        return described
