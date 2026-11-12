"""Training the router on what the modules have been used for."""

from __future__ import annotations

import shutil

import pytest

from gustav.inference.registry import load_registry
from gustav.inference.router import KeywordScorer, Router
from gustav.learn.examples import Example, ExampleStore
from gustav.optimize.router_fit import (
    RouterModel,
    agreement_with_keywords,
    calibrate,
    fit,
    fit_weights,
    module_documents,
)


@pytest.fixture
def registry(project, repo_root):
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")
    return load_registry(project / "modules" / "registry.yaml")


def seed_material(db, *, module: str, prompts: list[str], repeat: int = 1) -> None:
    """Store material for a module.

    Prompts are made distinct: the example store deduplicates by content, so twenty
    copies of one sentence is one piece of material, not twenty.
    """
    store = ExampleStore(db, heldout_fraction=0.0)
    store.add_many(
        Example(
            prompt=f"{prompt} (case {index})",
            completion="an answer",
            kind="study",
            module=module,
        )
        for index in range(repeat)
        for prompt in prompts
    )


def test_a_module_scores_for_vocabulary_its_keyword_list_never_had(db, registry):
    """The whole point. 'axum' and 'extractor' are nowhere in the rust module's entry."""
    entry = registry.get("rust")
    assert "axum" not in entry.keywords and "axum" not in entry.description.lower()

    seed_material(
        db,
        module="rust",
        prompts=[
            "my axum handler needs an extractor for the request body",
            "axum rejects my extractor ordering",
            "how do I write an axum handler with state",
        ],
        repeat=8,
    )
    model, reason = fit(db, registry, keyword_threshold=0.45)
    assert model is not None, reason

    text = "my axum handler needs an extractor"
    learned = model.score(text, registry.modules)
    keyword = KeywordScorer(registry.modules).score(text, registry.modules)
    assert keyword["rust"] == 0.0, "the keyword list cannot see this request at all"
    assert learned["rust"] > 0.3, learned


def test_a_word_every_module_uses_carries_little_weight(db, registry):
    documents = {
        "rust": ["lifetime lifetime borrow"] + ["please help"] * 3,
        "coding": ["regex regex compile"] + ["please help"] * 3,
    }
    weights = fit_weights(documents)
    assert weights["rust"]["lifetime"] > weights["rust"].get("please", 0.0)
    assert weights["coding"]["regex"] > weights["coding"].get("help", 0.0)


def test_scores_stay_between_zero_and_one(db, registry):
    seed_material(db, module="rust", prompts=["borrowck lifetime closure"], repeat=25)
    model, _ = fit(db, registry, keyword_threshold=0.45)
    for text in ("borrowck lifetime closure", "", "completely unrelated words here"):
        scores = model.score(text, registry.modules)
        assert all(0.0 <= value <= 1.0 for value in scores.values()), scores


def test_too_little_material_is_refused(db, registry):
    seed_material(db, module="rust", prompts=["one prompt"])
    model, reason = fit(db, registry, keyword_threshold=0.45)
    assert model is None
    assert "before a fitted router" in reason


def test_a_modules_description_anchors_it_even_with_no_material(db, registry):
    documents = module_documents(db, registry)
    assert all(len(texts) >= 3 for texts in documents.values())
    weights = fit_weights(documents)
    # "rust" is in the rust module's description, so it scores for it from the start.
    assert weights["rust"].get("rust", 0.0) > 0


def test_the_threshold_is_calibrated_to_keep_the_uncovered_rate(db, registry):
    requests = [
        "how do I use lifetimes in rust",
        "what does cargo clippy do",
        "explain a python traceback",
        "what is the capital of france",
        "sing me a song about geese",
    ]
    seed_material(db, module="rust", prompts=["lifetime borrow cargo clippy"], repeat=25)
    model, _ = fit(db, registry, keyword_threshold=0.45)
    threshold, before, after = calibrate(
        model, requests, registry, keyword_threshold=0.45
    )
    assert 0.0 < threshold <= 1.0
    assert abs(after - before) <= 0.25, (before, after)


def test_calibration_with_no_history_keeps_the_configured_threshold(db, registry):
    model = RouterModel(weights={}, threshold=0.9)
    threshold, before, after = calibrate(model, [], registry, keyword_threshold=0.45)
    assert (threshold, before, after) == (0.45, 0.0, 0.0)


def test_agreement_is_only_measured_where_the_keyword_scorer_was_confident(db, registry):
    seed_material(db, module="rust", prompts=["lifetime borrow cargo clippy rustc"], repeat=25)
    model, _ = fit(db, registry, keyword_threshold=0.45)
    # Nothing a module covers: nothing to agree about, so the answer is None, not 0.
    assert agreement_with_keywords(
        model, ["mmm", "hello there"], registry, keyword_threshold=0.9
    ) is None
    score = agreement_with_keywords(
        model,
        ["lifetime borrow", "cargo clippy rustc"],
        registry,
        keyword_threshold=0.2,
    )
    assert score is not None and 0.0 <= score <= 1.0


def test_a_fitted_model_round_trips_through_disk(db, registry, tmp_path):
    seed_material(db, module="rust", prompts=["borrowck lifetime closure"], repeat=25)
    model, _ = fit(db, registry, keyword_threshold=0.45)
    model.save(tmp_path)
    again = RouterModel.load(tmp_path)
    assert again is not None
    text = "borrowck lifetime"
    assert again.score(text, registry.modules) == model.score(text, registry.modules)
    assert again.threshold == model.threshold


def test_a_missing_or_broken_model_file_loads_as_none(tmp_path):
    assert RouterModel.load(tmp_path) is None
    (tmp_path / "model.json").write_text("not json", encoding="utf-8")
    assert RouterModel.load(tmp_path) is None


def test_the_router_can_be_driven_by_a_fitted_model(db, registry):
    seed_material(db, module="rust", prompts=["borrowck lifetime closure capture"], repeat=25)
    model, _ = fit(db, registry, keyword_threshold=0.45)
    router = Router(registry, scorer=model, threshold=model.threshold)
    decision = router.route("borrowck rejected my closure capture")
    assert "rust" in decision.modules
    assert decision.threshold == model.threshold
