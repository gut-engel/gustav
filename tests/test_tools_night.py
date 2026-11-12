"""The night writing tools, and the review that follows."""

from __future__ import annotations

import json
import shutil

import pytest

from gustav.config.schema import Config
from gustav.sleep.stages import tools as tools_step
from gustav.testing.steps import ScriptedModel, make_env
from gustav.tools.store import ToolStore


def body(module):
    return module.run_step.__wrapped__


ADD_REPLY = json.dumps(
    {
        "name": "add_numbers",
        "description": "Add two numbers together and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "the first number"},
                "b": {"type": "number", "description": "the second number"},
            },
            "required": ["a", "b"],
        },
        "capabilities": ["pure"],
        "source": "def run(a, b):\n    return a + b",
        "examples": [{"arguments": {"a": 2, "b": 3}, "expect": 5}],
    }
)


@pytest.fixture
def env_for(project, paths, db, repo_root):
    shutil.copy(repo_root / "modules" / "registry.yaml", project / "modules" / "registry.yaml")

    def build(*, model=None, **overrides):
        data = {"logging": {"console": False}, "tools": {"timeout_seconds": 6}}
        data.update(overrides)
        return make_env(
            config=Config.model_validate(data), paths=paths, db=db, step="tools",
            model=model,
        )

    return build


def test_nothing_is_written_unless_something_asked(env_for):
    result = body(tools_step)(env_for(model=ScriptedModel([ADD_REPLY])))
    assert result["written"] == []
    assert "nothing asked for a tool" in result["reason"]


def test_a_request_is_written_and_left_for_review(env_for, db, paths):
    store = ToolStore(db, paths.tools)
    store.add_request("add two numbers together")
    env = env_for(model=ScriptedModel([ADD_REPLY]))

    result = body(tools_step)(env)
    assert [w["name"] for w in result["written"]] == ["add_numbers"]
    assert result["pending_review"] == 1
    assert store.approved() == [], "the night cannot approve its own tool"
    assert (paths.tools / "add_numbers" / "v001" / "tool.py").exists()
    assert store.open_requests() == [], "the request was answered"
    assert db.query_one("SELECT status FROM tool_requests")["status"] == "written"


def test_the_event_tells_you_it_needs_reading(env_for, db, paths):
    ToolStore(db, paths.tools).add_request("add two numbers")
    body(tools_step)(env_for(model=ScriptedModel([ADD_REPLY])))
    event = db.query_one("SELECT * FROM events WHERE kind = 'tool.proposed'")
    assert event["level"] == "warning"
    assert "until you approve it" in event["message"]
    assert "ai tools show add_numbers" in event["message"]


def test_a_tool_that_fails_its_checks_is_refused_and_the_request_says_why(
    env_for, db, paths
):
    store = ToolStore(db, paths.tools)
    store.add_request("do something impossible")
    vacuous = json.dumps(
        {
            "name": "does_nothing",
            "description": "A tool whose examples prove nothing whatsoever.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number", "description": "x"}},
                "required": ["a"],
            },
            "capabilities": ["pure"],
            "source": "def run(a):\n    return None",
            "examples": [{"arguments": {"a": 1}, "expect": None}],
        }
    )
    result = body(tools_step)(env_for(model=ScriptedModel([vacuous, vacuous])))

    assert result["written"] == []
    assert "do not test the tool" in result["refused"][0]["reason"]
    assert store.record("does_nothing") is None, "nothing reached the store"
    assert db.query_one("SELECT status, detail FROM tool_requests")["status"] == "refused"


def test_a_failed_task_that_looks_computational_becomes_a_tool(env_for, db, paths):
    db.insert(
        "tasks",
        {
            "module": "coding", "kind": "python", "prompt": "convert celsius to fahrenheit",
            "verifier": "{}", "attempts": 3, "passes": 0, "fingerprint": "f1",
        },
    )
    result = body(tools_step)(env_for(model=ScriptedModel([ADD_REPLY])))
    assert result["requests"] == 1
    assert [w["name"] for w in result["written"]] == ["add_numbers"]


def test_a_failed_task_that_training_would_fix_is_left_alone(env_for, db):
    db.insert(
        "tasks",
        {
            "module": "general", "kind": "python", "prompt": "explain why the sky is blue",
            "verifier": "{}", "attempts": 3, "passes": 0, "fingerprint": "f2",
        },
    )
    result = body(tools_step)(env_for(model=ScriptedModel([ADD_REPLY])))
    assert result["written"] == [] and result["requests"] == 0


def test_authoring_can_be_switched_off(env_for, db, paths):
    ToolStore(db, paths.tools).add_request("add two numbers")
    result = body(tools_step)(env_for(model=ScriptedModel([ADD_REPLY]),
                                      tools={"allow_authoring": False}))
    assert "allow_authoring is false" in result["reason"]


def test_without_a_model_it_says_so(env_for, db, paths):
    ToolStore(db, paths.tools).add_request("add two numbers")
    result = body(tools_step)(env_for())
    assert result["written"] == [] and "model" in result["reason"]


def test_a_dry_run_writes_nothing(env_for, db, paths):
    ToolStore(db, paths.tools).add_request("add two numbers")
    env = env_for(model=ScriptedModel([ADD_REPLY]))
    env.dry_run = True
    result = body(tools_step)(env)
    assert result["dry_run"] and result["written"] == []
    assert not (paths.tools / "add_numbers").exists()


def test_the_per_night_limit_is_respected(env_for, db, paths):
    store = ToolStore(db, paths.tools)
    for text in ("add two numbers", "subtract two numbers", "multiply two numbers"):
        store.add_request(text)
    env = env_for(model=ScriptedModel([ADD_REPLY]), tools={"max_new_per_run": 1,
                                                           "timeout_seconds": 6})
    result = body(tools_step)(env)
    assert result["requests"] == 1
    assert len(store.open_requests()) == 2
