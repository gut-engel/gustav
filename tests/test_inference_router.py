"""The module registry and the router."""

from __future__ import annotations

import pytest

from gustav.inference.registry import ModuleSpec, Registry, load_registry
from gustav.inference.router import KeywordScorer, Router

pytestmark = []


@pytest.fixture
def registry(repo_root):
    return load_registry(repo_root / "modules" / "registry.yaml")


def test_the_shipped_registry_loads(registry):
    assert "coding" in registry and "rust" in registry
    assert [m.name for m in registry.always_on] == ["general", "personal-memory"]


def test_unknown_module_lists_the_known_ones(registry):
    with pytest.raises(KeyError, match="coding"):
        registry.get("cobol")


def test_a_typo_in_the_registry_is_an_error(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("modules:\n  - name: x\n    keywordz: [a]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keywordz"):
        load_registry(path)


def test_duplicate_module_names_are_an_error(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("modules:\n  - name: x\n  - name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(path)


def test_a_missing_registry_is_empty_not_an_error(tmp_path):
    assert len(load_registry(tmp_path / "nope.yaml")) == 0


def test_a_module_can_be_added_and_written_back(registry, tmp_path):
    grown = registry.add(
        ModuleSpec(name="sql", description="databases and queries", keywords=("select", "join"))
    )
    assert "sql" in grown
    path = grown.write(tmp_path / "registry.yaml")
    assert "sql" in load_registry(path)


def test_adding_an_existing_module_is_refused(registry):
    with pytest.raises(ValueError, match="already exists"):
        registry.add(ModuleSpec(name="coding"))


# --- routing --------------------------------------------------------------
def router(registry, **kwargs) -> Router:
    return Router(registry, **kwargs)


def test_a_rust_question_routes_to_rust(registry):
    decision = router(registry).route("why does the borrow checker reject this &mut?")
    assert "rust" in decision.modules
    assert decision.confidence > decision.threshold


def test_a_python_question_routes_to_coding_not_rust(registry):
    decision = router(registry).route("write a python function to parse json")
    assert "coding" in decision.modules and "rust" not in decision.modules


def test_always_on_modules_are_always_included(registry):
    decision = router(registry).route("anything at all")
    assert "general" in decision.modules and "personal-memory" in decision.modules


def test_an_everyday_question_is_not_flagged_as_uncovered(registry):
    """An always-on module scoring well is coverage, not a gap."""
    decision = router(registry).route("summarise this email for me")
    assert not decision.uncovered


def test_a_topic_no_module_covers_is_flagged(registry):
    """This is one of the self-expansion triggers in section 8."""
    decision = router(registry).route("what is the capital of peru")
    assert decision.uncovered


def test_max_active_limits_the_stack(registry):
    decision = router(registry, max_active=2).route(
        "rust cargo crate trait lifetime borrow python code error api"
    )
    assert len(decision.modules) <= 2
    assert decision.dropped


def test_higher_priority_modules_come_first(registry):
    decision = router(registry).route("my rust borrow checker problem")
    assert decision.modules[0] == "personal-memory"  # priority 10


def test_explicit_modules_override_the_router(registry):
    decision = router(registry).route("anything", forced=("rust",))
    assert decision.modules == ("rust",) and decision.confidence == 1.0


def test_unknown_forced_modules_are_dropped(registry):
    decision = router(registry).route("anything", forced=("rust", "cobol"))
    assert decision.modules == ("rust",)


def test_only_available_modules_are_routed_to(registry):
    decision = router(registry, available={"general"}).route("rust borrow checker lifetime")
    assert decision.modules == ("general",)


def test_no_available_modules_routes_to_nothing(registry):
    decision = router(registry, available=set()).route("rust")
    assert decision.modules == ()
    assert "no trained modules" in decision.reason


def test_a_higher_threshold_routes_more_narrowly(registry):
    loose = router(registry, threshold=0.05).route("a cargo question")
    strict = router(registry, threshold=0.99).route("a cargo question")
    assert len(loose.modules) >= len(strict.modules)


def test_the_decision_is_explainable(registry):
    decision = router(registry).route("rust lifetimes")
    assert "threshold" in decision.describe()
    assert set(decision.scores) == set(registry.names)


def test_idf_makes_a_shared_word_worthless(registry):
    scorer = KeywordScorer(registry.modules)
    scores = scorer.score("explain", registry.modules)
    # "explain" appears in the general module only, so it should favour it.
    assert scores["general"] >= max(scores[m] for m in ("coding", "rust"))


def test_empty_input_scores_nothing(registry):
    decision = router(registry).route("")
    assert decision.confidence == 0.0


def test_decisions_are_logged_for_later_training(registry, db):
    from gustav.inference.router import RouterLog

    log = RouterLog(db)
    log.record(router(registry).route("rust lifetimes"), turn_id=7)
    row = db.query_one("SELECT metric, tags FROM telemetry ORDER BY id DESC LIMIT 1")
    assert row["metric"] == "router.decision"
    assert "rust" in row["tags"]
