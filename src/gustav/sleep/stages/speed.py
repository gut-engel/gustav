"""Speed: make serving faster, without making it worse.

Two jobs, both measured.

**The router.** Stage 3's keyword scorer matches a request against hand-written module
descriptions, so a question that uses none of the listed keywords scores zero and is
recorded as uncovered. This fits a scorer from the vocabulary each module has actually
accumulated and installs it only if it still agrees with the keyword scorer where that
one was confident, and only if its calibrated threshold does not suddenly change how
many requests are called uncovered — which would otherwise flood self-expansion with
false triggers.

**The tunables.** Section 10: the daemon may change the values in ``tunable`` and
nothing else. Each trial reloads the model with one value changed, measures tokens per
second on a short fixed benchmark, and — for anything that can change what the model
*says* — scores a slice of the held-out suite as well. A win has to be a win on speed
*and* not a loss on quality, or the value goes back and is remembered as a loser.

**Speculative decoding.** The generation loop can use a small model to propose tokens
that the real one verifies, which for greedy decoding is exact — the accepted tokens are
the ones the target model would have produced anyway. The step reports whether a draft
model is available and what the acceptance rate has been; turning it on or off is one of
the tunable settings above, so it is kept or dropped on a measurement like everything
else. It does not *distil* a draft model: training a small model well enough to be worth
proposing is a project, not a night, and the ladder already contains models small enough
to serve as drafts.
"""

from __future__ import annotations

from typing import Any

from gustav.inference.registry import load_registry
from gustav.optimize.router_fit import fit
from gustav.optimize.tuner import QUALITY_SENSITIVE, Measurement, Tuner
from gustav.sleep.stages._base import StepEnv, sleep_step

#: A fitted router that disagrees with the keyword scorer this often, on the requests
#: the keyword scorer was sure about, is not an improvement — it is a different opinion
#: with no evidence behind it.
MIN_AGREEMENT = 0.6
#: And its uncovered rate may not move by more than this, in absolute terms.
MAX_UNCOVERED_SHIFT = 0.15
#: Held-out items scored per trial, when a setting can change what is generated.
QUALITY_ITEMS = 8


def _router_dir(env: StepEnv):
    return env.paths.data_dir / "router"


def train_router(env: StepEnv) -> dict[str, Any]:
    registry = load_registry(env.paths.registry)
    model, reason = fit(
        env.db, registry, keyword_threshold=env.config.tunable.router_threshold
    )
    outcome: dict[str, Any] = {"installed": False, "reason": reason}
    if model is None:
        return outcome

    outcome.update(
        {
            "threshold": model.threshold,
            "agreement": model.agreement,
            "uncovered_before": model.uncovered_before,
            "uncovered_after": model.uncovered_after,
            "modules": len(model.modules),
            "material": model.documents,
        }
    )
    if model.agreement is not None and model.agreement < MIN_AGREEMENT:
        outcome["reason"] = (
            f"the fitted router agrees with the keyword scorer on only "
            f"{model.agreement:.0%} of the requests it was confident about; "
            f"{MIN_AGREEMENT:.0%} is the floor"
        )
        return outcome
    shift = abs((model.uncovered_after or 0.0) - (model.uncovered_before or 0.0))
    if shift > MAX_UNCOVERED_SHIFT:
        outcome["reason"] = (
            f"it would change the share of requests called uncovered by "
            f"{shift:.0%}, over the {MAX_UNCOVERED_SHIFT:.0%} allowed"
        )
        return outcome
    if env.dry_run:
        outcome["reason"] = "dry run: the fitted router was not installed"
        return outcome

    path = model.save(_router_dir(env))
    outcome["installed"] = True
    outcome["path"] = str(path)
    outcome["reason"] = reason
    agreement = (
        f"{model.agreement:.0%} agreement" if model.agreement is not None
        else "no confident requests to compare against"
    )
    env.events.emit(
        "speed.router",
        f"router refitted on {sum(model.documents.values())} pieces of material "
        f"({agreement}, threshold {model.threshold:.2f})",
        run_id=env.run_id,
        agreement=model.agreement,
        threshold=model.threshold,
    )
    return outcome


def _quality_prompts(env: StepEnv, limit: int = QUALITY_ITEMS) -> list[tuple[str, str]]:
    """(prompt, expected) pairs from the held-out split, for the quality check."""
    rows = env.db.query(
        "SELECT prompt, completion FROM training_examples WHERE split = 'heldout'"
        " ORDER BY fingerprint LIMIT ?",
        (int(limit),),
    )
    return [(row["prompt"], row["completion"]) for row in rows]


def tune(env: StepEnv) -> dict[str, Any]:
    from gustav.config.load import pinned_tunables

    cfg = env.config.optimisation
    outcome: dict[str, Any] = {"trials": [], "kept": {}, "skipped": []}
    if not cfg.enabled:
        outcome["reason"] = "optimisation.enabled is false"
        return outcome

    tuner = Tuner(
        config=cfg,
        tunable=env.config.tunable,
        paths=env.paths,
        db=env.db,
        events=env.events,
        pinned=pinned_tunables(env.paths.root / "config.yaml"),
    )
    pairs = tuner.candidates()
    outcome["candidates"] = [f"{setting}={value!r}" for setting, value in pairs]
    outcome["skipped"] = tuner.why_skipped()
    if not pairs:
        outcome["reason"] = "every allowed setting has already been measured"
        return outcome
    if env.dry_run:
        outcome["dry_run"] = True
        return outcome

    quality_items = _quality_prompts(env)

    def measure(overrides: dict[str, Any]) -> Measurement:
        """Load the model under one changed setting and measure it.

        The override is applied, measured and *undone* here. Only the tuner may make a
        value stick, and only after it has judged the trial — otherwise a losing value
        would be left in place by the measurement that rejected it.
        """
        previous = {
            setting: getattr(env.config.tunable, setting) for setting in overrides
        }
        for setting, value in overrides.items():
            setattr(env.config.tunable, setting, value)
        # A fresh handle per trial: `prefix_cache` and the KV dtype are decided when
        # the backend is built, so re-using a loaded model would measure the old value.
        model = env.new_model()
        try:
            if not model.available:
                raise RuntimeError(model.error or "the model would not load")
            speed = model.benchmark(
                max_tokens=cfg.bench_tokens, repeats=cfg.bench_repeats
            )
            quality = None
            if quality_items and (not overrides or set(overrides) & QUALITY_SENSITIVE):
                quality = model.quality(quality_items)
            return Measurement(
                tokens_per_second=speed["tokens_per_second"],
                quality=quality,
                detail=speed,
            )
        finally:
            model.close()
            for setting, value in previous.items():
                setattr(env.config.tunable, setting, value)

    result = tuner.run(
        measure, run_id=env.run_id, progress=lambda message: env.progress(message)
    )
    outcome["trials"] = [t.describe() for t in result.trials]
    outcome["skipped"] = result.skipped
    outcome["kept"] = {k: repr(v) for k, v in result.kept.items()}
    if result.baseline:
        outcome["baseline_tokens_per_second"] = round(result.baseline.tokens_per_second, 2)
    outcome["summary"] = result.summary()
    return outcome


def draft_model_state(env: StepEnv) -> dict[str, Any]:
    """Where speculative decoding stands: is there a draft, and is it being accepted?"""
    from gustav.hw.service import HardwareService
    from gustav.inference.transformers_backend import resolve_model_path

    hardware = HardwareService(env.config, env.paths, env.db, env.events)
    plan = hardware.stored_plan()
    if plan is None:
        return {"supported": False, "reason": "there is no hardware plan"}

    wanted = bool(plan.speculative_decoding or env.config.tunable.speculative_decoding)
    ladder = hardware.effective_ladder()
    draft_id = plan.draft_model
    if draft_id is None:
        try:
            draft = ladder.draft_for(ladder.model(plan.base_model))
        except KeyError:
            draft = None
        draft_id = draft.id if draft is not None else None

    state: dict[str, Any] = {
        "wanted": wanted,
        "draft_model": draft_id,
        "lookahead": 4,
        "on_disk": False,
        "supported": False,
    }
    # What acceptance has looked like, if a benchmark has ever run with it on. Read
    # before the availability checks: "it used to accept 62% and now there is no draft
    # model on disk" is exactly the situation worth seeing.
    row = env.db.query_one(
        "SELECT value FROM telemetry WHERE metric = 'speculation.acceptance'"
        " ORDER BY id DESC LIMIT 1"
    )
    if row is not None:
        state["acceptance"] = round(float(row["value"]), 3)

    if draft_id is None:
        state["reason"] = "no model on the ladder is small enough to be a draft"
        return state
    try:
        resolve_model_path(env.paths.models, ladder.model(draft_id))
        state["on_disk"] = True
    except Exception as exc:  # noqa: BLE001 - "not downloaded" is the normal case
        state["reason"] = f"{draft_id} is not on disk: ai model pull {draft_id}"
        state["detail"] = str(exc)[:200]
        return state

    state["supported"] = True
    state["reason"] = (
        f"{draft_id} is available as a draft model"
        + (
            "; speculative decoding is on"
            if wanted
            else "; set tunable.speculative_decoding to try it, or let the night measure it"
        )
    )
    return state


@sleep_step
def run_step(env: StepEnv) -> dict[str, Any]:
    result: dict[str, Any] = {}
    env.progress("fitting the router")
    result["router"] = train_router(env)
    env.progress("measuring tunable settings")
    result["optimisation"] = tune(env)
    result["draft"] = draft_model_state(env)
    env.write_json("speed.json", result)
    return result
