"""Tools the model writes: the sandbox, the admission checks, the store, the review."""

from __future__ import annotations

import json

import pytest

from gustav.config.schema import ToolsConfig
from gustav.tools.policy import CAPABILITIES, policy_for, unsupported
from gustav.tools.review import check
from gustav.tools.runner import ToolRunner, _last_json
from gustav.tools.spec import (
    ToolSpec,
    imports_of,
    signature_of,
    stub_of,
    validate_name,
    validate_schema,
)
from gustav.tools.store import ToolError, ToolStore


@pytest.fixture
def config():
    # Short timeouts: these tests should not take seconds each.
    return ToolsConfig(timeout_seconds=6, memory_limit_mb=512)


@pytest.fixture
def runner(tmp_path):
    return ToolRunner(workspace_root=tmp_path)


def spec_for(source: str, **overrides) -> ToolSpec:
    defaults = {
        "name": "add_numbers",
        "description": "Add two numbers together and return the sum.",
        "source": source,
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "the first number"},
                "b": {"type": "number", "description": "the second number"},
            },
            "required": ["a", "b"],
        },
        "examples": ({"arguments": {"a": 2, "b": 3}, "expect": 5},),
    }
    defaults.update(overrides)
    return ToolSpec(**defaults)


ADD = "def run(a, b):\n    return a + b"


# --- the sandbox --------------------------------------------------------
def test_a_tool_is_a_function_from_json_to_json(runner, config):
    result = runner.run(ADD, {"a": 2, "b": 3}, policy_for(["pure"], config))
    assert result.ok and result.result == 5
    assert result.for_model() == "5"


def test_arguments_it_does_not_accept_are_an_error_not_a_crash(runner, config):
    result = runner.run(ADD, {"a": 1, "zzz": 2}, policy_for(["pure"], config))
    assert not result.ok and result.kind == "arguments"
    assert "unexpected keyword argument" in result.error


def test_a_tool_that_raises_is_reported_with_its_message(runner, config):
    result = runner.run(
        "def run():\n    raise ValueError('bad input')", {}, policy_for(["pure"], config)
    )
    assert not result.ok and result.kind == "raised"
    assert "ValueError: bad input" in result.error
    assert "run" in result.detail


def test_source_that_does_not_compile_is_caught(runner, config):
    result = runner.run("def run(:\n  pass", {}, policy_for(["pure"], config))
    assert not result.ok and result.kind == "syntax"


def test_a_tool_with_no_run_function_is_refused(runner, config):
    result = runner.run("x = 1", {}, policy_for(["pure"], config))
    assert not result.ok and result.kind == "contract"


def test_a_result_json_cannot_represent_is_refused(runner, config):
    result = runner.run("def run():\n    return object()", {}, policy_for(["pure"], config))
    assert not result.ok and result.kind == "result"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import subprocess\ndef run():\n    return 1", "may not import"),
        ("import socket\ndef run():\n    return 1", "may not import"),
        ("import ctypes\ndef run():\n    return 1", "may not import"),
        # `os` itself is importable — `pathlib` and half the standard library import
        # it — so what is refused is what it would be used *for*.
        ("import os\ndef run():\n    return os.listdir('/')", "not listable"),
        ("import os\ndef run():\n    return os.system('echo hi')", "may not start processes"),
        ("def run():\n    return __import__('socket')", "may not import"),
        ("def run():\n    return open('/etc/hostname').read()", "not allowed"),
        ("def run():\n    open('out.txt', 'w').write('x')\n    return 1", "not allowed"),
    ],
)
def test_the_sandbox_refuses_what_a_pure_tool_must_not_do(runner, config, source, expected):
    result = runner.run(source, {}, policy_for(["pure"], config))
    assert not result.ok, result.result
    assert expected in result.error


def test_an_allowed_module_works(runner, config):
    result = runner.run(
        "import math\ndef run(x):\n    return math.floor(x)",
        {"x": 2.7},
        policy_for(["pure"], config),
    )
    assert result.ok and result.result == 2


def test_the_workspace_capability_allows_its_own_directory_only(runner, config):
    policy = policy_for(["pure", "write_workspace"], config)
    inside = runner.run(
        "def run():\n    open('note.txt', 'w').write('hi')\n    return open('note.txt').read()",
        {},
        policy,
    )
    assert inside.ok and inside.result == "hi"
    outside = runner.run(
        "def run():\n    open('/tmp/gustav-escape.txt', 'w').write('x')\n    return 1",
        {},
        policy,
    )
    assert not outside.ok and "not allowed" in outside.error


def test_each_call_gets_a_fresh_workspace(runner, config):
    policy = policy_for(["pure", "write_workspace"], config)
    runner.run("def run():\n    open('state.txt','w').write('1')\n    return 1", {}, policy)
    second = runner.run(
        "def run():\n    import pathlib\n    return pathlib.Path('state.txt').exists()",
        {},
        policy,
    )
    assert second.ok and second.result is False, "a tool cannot keep state between calls"


def test_an_endless_tool_is_killed(runner):
    config = ToolsConfig(timeout_seconds=2)
    result = runner.run("def run():\n    while True: pass", {}, policy_for(["pure"], config))
    assert not result.ok and result.timed_out and result.duration_ms < 20_000


def test_a_memory_hungry_tool_hits_its_limit(runner):
    config = ToolsConfig(timeout_seconds=10, memory_limit_mb=256)
    result = runner.run(
        "def run():\n    return len(bytearray(4_000_000_000))", {}, policy_for(["pure"], config)
    )
    assert not result.ok


def test_the_network_is_refused_when_no_hosts_are_allowed(runner, config):
    result = runner.run(
        "import urllib.parse\ndef run():\n    return urllib.parse.quote('x')",
        {},
        policy_for(["pure"], config),
    )
    assert result.ok, "parsing a URL is not using the network"
    assert not policy_for(["network"], config).may_network


def test_a_tool_cannot_raise_its_own_limits(runner, config):
    escape = (
        "def run():\n"
        "    import resource\n"
        "    soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (hard * 4, hard * 4))\n"
        "    return 'escaped'\n"
    )
    result = runner.run(escape, {}, policy_for(["pure"], config))
    assert not result.ok


def test_output_before_the_result_does_not_break_the_protocol(runner, config):
    result = runner.run(
        "def run():\n    print('working...')\n    return 7", {}, policy_for(["pure"], config)
    )
    assert result.ok and result.result == 7


def test_the_result_parser_is_robust():
    assert _last_json('{"ok": true, "result": 1}')["result"] == 1
    assert _last_json('noise\n{"ok": true, "result": 2}')["result"] == 2
    assert _last_json("") is None
    assert _last_json("not json") is None


# --- static checks ------------------------------------------------------
def test_the_signature_is_read_from_the_source_without_running_it():
    assert [(p.name, p.required) for p in signature_of(ADD)] == [("a", True), ("b", True)]
    assert signature_of("def run(**kw):\n    pass") is None, "**kwargs is unverifiable"
    assert signature_of("x = 1") is None
    assert imports_of("import math\nfrom re import sub") == {"math", "re"}


def test_the_stub_keeps_the_signature_and_loses_the_body():
    assert stub_of(ADD) == "def run(a, b):\n    return None\n"


def test_a_schema_that_does_not_match_the_signature_is_refused():
    problems = validate_schema(
        {"type": "object", "properties": {"a": {"type": "number", "description": "x"}},
         "required": ["a"]},
        ADD,
    )
    assert any("run() takes 'b'" in p for p in problems)


def test_tool_names_are_restricted():
    assert validate_name("convert_units") is None
    for bad in ("Convert", "9lives", "x", "with spaces", ""):
        assert validate_name(bad) is not None


# --- admission ----------------------------------------------------------
def test_a_sound_tool_passes_every_check(config, runner):
    checks = check(spec_for(ADD), config, runner=runner)
    assert checks.ok, checks.problems
    assert checks.examples[0].ok and checks.examples[0].got == 5
    assert not checks.stub_passes


def test_examples_that_do_not_match_are_refused(config, runner):
    spec = spec_for(ADD, examples=({"arguments": {"a": 2, "b": 3}, "expect": 6},))
    checks = check(spec, config, runner=runner)
    assert not checks.ok
    assert "did not produce what the tool claims" in " ".join(checks.problems)


def test_examples_that_assert_nothing_are_refused(config, runner):
    spec = spec_for(
        "def run(a, b):\n    return None",
        examples=({"arguments": {"a": 1, "b": 2}, "expect": None},),
    )
    checks = check(spec, config, runner=runner)
    assert checks.stub_passes
    assert "do not test the tool at all" in " ".join(checks.problems)


def test_a_tool_with_no_examples_is_refused(config, runner):
    checks = check(spec_for(ADD, examples=()), config, runner=runner)
    assert "comes with no examples" in " ".join(checks.problems)


def test_capabilities_beyond_the_allowance_are_refused(config, runner):
    spec = spec_for(ADD, capabilities=("network",))
    checks = check(spec, config, runner=runner)
    assert "does not allow" in " ".join(checks.problems)
    assert unsupported(["network"], config.allow_capabilities) == ["network"]
    assert unsupported(["pure"], config.allow_capabilities) == []


def test_an_unknown_capability_is_refused(config, runner):
    checks = check(spec_for(ADD, capabilities=("telepathy",)), config, runner=runner)
    assert "does not allow" in " ".join(checks.problems)
    assert "telepathy" not in CAPABILITIES


def test_a_forbidden_import_is_refused_before_anything_runs(config, runner):
    spec = spec_for("import subprocess\n" + ADD)
    checks = check(spec, config, runner=runner)
    assert "may never use" in " ".join(checks.problems)
    assert checks.examples == [], "the examples are not even run"


def test_a_description_too_short_to_route_on_is_refused(config, runner):
    checks = check(spec_for(ADD, description="adds"), config, runner=runner)
    assert "too short" in " ".join(checks.problems)


# --- the store ----------------------------------------------------------
@pytest.fixture
def store(db, paths):
    return ToolStore(db, paths.tools)


def test_a_proposed_tool_is_on_disk_and_not_callable(store, config, runner, paths):
    spec = spec_for(ADD)
    version = store.propose(spec, check(spec, config, runner=runner))

    assert version.status == "proposed"
    assert (paths.tools / "add_numbers" / "v001" / "tool.py").read_text() == ADD
    meta = json.loads((paths.tools / "add_numbers" / "v001" / "meta.json").read_text())
    assert meta["capabilities"] == ["pure"] and meta["status"] == "proposed"
    assert store.approved() == [], "nothing is callable before it is read"
    assert [v.name for v in store.pending()] == ["add_numbers"]


def test_approval_makes_it_callable_and_links_it(store, config, runner, paths):
    spec = spec_for(ADD)
    store.propose(spec, check(spec, config, runner=runner))
    store.approve("add_numbers", by="simon", note="read it, fine")

    approved = store.approved()
    assert [v.name for v in approved] == ["add_numbers"]
    assert (paths.tools / "add_numbers" / "active").readlink().name == "v001"
    assert store.record("add_numbers").callable
    assert store.version("add_numbers").reviewed_by == "simon"


def test_a_tool_that_failed_its_checks_cannot_be_approved(store, config, runner):
    spec = spec_for(
        "def run(a, b):\n    return None",
        examples=({"arguments": {"a": 1, "b": 1}, "expect": None},),
    )
    store.propose(spec, check(spec, config, runner=runner))
    with pytest.raises(ToolError, match="did not pass its admission checks"):
        store.approve("add_numbers")


def test_rejecting_leaves_the_source_and_the_reason(store, config, runner, paths):
    spec = spec_for(ADD)
    store.propose(spec, check(spec, config, runner=runner))
    store.reject("add_numbers", by="simon", note="I do not want this")

    assert store.approved() == []
    assert store.record("add_numbers").status == "rejected"
    assert (paths.tools / "add_numbers" / "v001" / "tool.py").exists()
    meta = json.loads((paths.tools / "add_numbers" / "v001" / "meta.json").read_text())
    assert meta["status"] == "rejected"


def test_a_new_version_does_not_disturb_the_approved_one(store, config, runner):
    spec = spec_for(ADD)
    store.propose(spec, check(spec, config, runner=runner))
    store.approve("add_numbers")

    better = spec_for(
        "def run(a, b):\n    return float(a) + float(b)",
        examples=({"arguments": {"a": 2, "b": 3}, "expect": 5.0},),
    )
    store.propose(better, check(better, config, runner=runner))

    assert [v.version for v in store.approved()] == ["v001"], "v002 awaits its own review"
    assert [v.status for v in store.versions("add_numbers")] == ["approved", "proposed"]

    store.approve("add_numbers", "v002")
    assert [v.version for v in store.approved()] == ["v002"]
    assert store.versions("add_numbers")[0].status == "superseded"


def test_a_tool_can_be_rolled_back_to_the_previous_version(store, config, runner):
    first = spec_for(ADD)
    store.propose(first, check(first, config, runner=runner))
    store.approve("add_numbers")
    second = spec_for(
        "def run(a, b):\n    return float(a) + float(b)",
        examples=({"arguments": {"a": 2, "b": 3}, "expect": 5.0},),
    )
    store.propose(second, check(second, config, runner=runner))
    store.approve("add_numbers", "v002")

    rolled = store.rollback("add_numbers")
    assert rolled.version == "v001"
    assert [v.version for v in store.approved()] == ["v001"]


def test_retiring_stops_offering_it_without_deleting_it(store, config, runner, paths):
    spec = spec_for(ADD)
    store.propose(spec, check(spec, config, runner=runner))
    store.approve("add_numbers")
    store.retire("add_numbers")

    assert store.approved() == []
    assert not (paths.tools / "add_numbers" / "active").exists()
    assert (paths.tools / "add_numbers" / "v001" / "tool.py").exists()


def test_the_tool_limit_is_enforced(db, paths, config, runner):
    store = ToolStore(db, paths.tools, max_tools=1)
    first = spec_for(ADD)
    store.propose(first, check(first, config, runner=runner))
    other = spec_for(ADD, name="other_tool")
    with pytest.raises(ToolError, match="max_tools"):
        store.propose(other, check(other, config, runner=runner))


def test_proposing_the_same_source_twice_is_refused(store, config, runner):
    spec = spec_for(ADD)
    store.propose(spec, check(spec, config, runner=runner))
    with pytest.raises(ToolError, match="proposed before"):
        store.propose(spec, check(spec, config, runner=runner))


def test_calls_are_logged_with_their_arguments(store, config, runner):
    spec = spec_for(ADD)
    store.propose(spec, check(spec, config, runner=runner))
    store.record_call(
        "add_numbers", version="v001", arguments={"a": 1, "b": 2}, result=3, ok=True,
        duration_ms=12, session_id=4,
    )
    store.record_call(
        "add_numbers", version="v001", arguments={"a": "x"}, result=None, ok=False,
        error="TypeError", duration_ms=8,
    )
    assert store.usage("add_numbers") == {
        "calls": 2, "ok": 1, "failed": 1, "last": store.calls()[0]["created_at"],
    }
    assert json.loads(store.calls()[1]["arguments"]) == {"a": 1, "b": 2}


def test_requests_are_queued_and_closed(store):
    request_id = store.add_request("convert between units")
    assert [r["text"] for r in store.open_requests()] == ["convert between units"]
    store.close_request(request_id, status="refused", detail="could not write it")
    assert store.open_requests() == []
    assert store.withdraw_request(request_id) is False


# --- the night writes them ----------------------------------------------
def test_the_model_is_told_the_rules_it_will_be_judged_by(config):
    from gustav.tools.author import system_prompt

    prompt = system_prompt(config)
    assert "run(" in prompt
    for capability in config.allow_capabilities:
        assert capability in prompt
    assert "math" in prompt, "the module allowlist is in the prompt"
    assert "emptied out" in prompt, "it is told its examples must test something"


def test_a_well_written_tool_is_accepted_first_time(config, runner):
    from gustav.testing.steps import ScriptedModel
    from gustav.tools.author import write_tool

    reply = json.dumps(
        {
            "name": "add_numbers",
            "description": "Add two numbers together and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "the first"},
                    "b": {"type": "number", "description": "the second"},
                },
                "required": ["a", "b"],
            },
            "capabilities": ["pure"],
            "source": ADD,
            "examples": [{"arguments": {"a": 2, "b": 3}, "expect": 5}],
        }
    )
    authored = write_tool(ScriptedModel([reply]), "add numbers", config, runner=runner)
    assert authored.ok and authored.attempts == 1
    assert authored.spec.name == "add_numbers"


def test_the_checks_are_handed_back_so_it_can_repair_its_own_work(config, runner):
    from gustav.testing.steps import ScriptedModel
    from gustav.tools.author import write_tool

    broken = json.dumps(
        {
            "name": "add_numbers",
            "description": "Add two numbers together and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number", "description": "the first"}},
                "required": ["a"],
            },
            "capabilities": ["pure"],
            "source": ADD,
            "examples": [{"arguments": {"a": 2, "b": 3}, "expect": 5}],
        }
    )
    fixed = json.loads(broken)
    fixed["parameters"]["properties"]["b"] = {"type": "number", "description": "the second"}
    fixed["parameters"]["required"] = ["a", "b"]

    model = ScriptedModel([broken, json.dumps(fixed)])
    authored = write_tool(model, "add numbers", config, runner=runner)

    assert authored.ok and authored.attempts == 2
    assert "run() takes 'b'" in model.prompts[1], "it was told what was wrong"


def test_two_bad_attempts_are_reported_not_retried_forever(config, runner):
    from gustav.testing.steps import ScriptedModel
    from gustav.tools.author import write_tool

    model = ScriptedModel(["not json", "still not json"])
    authored = write_tool(model, "anything", config, runner=runner)
    assert not authored.ok
    assert model.calls == 2
    assert "JSON" in authored.reason
