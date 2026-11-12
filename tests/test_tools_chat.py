"""Calling tools during a conversation, with no GPU and no weights.

The fake backend is scripted to emit a tool call the way a chat template would, which
is enough to exercise the part that matters: the model asks, the sandbox runs it, the
result goes back as a tool message, and the second round of generation sees it.
"""

from __future__ import annotations

import json
import shutil

import pytest
import pytest_asyncio

from gustav.config.schema import Config
from gustav.core.events import Events, Telemetry
from gustav.hw.detect import simulated
from gustav.hw.ladder import load_ladder
from gustav.hw.service import HardwareService
from gustav.inference.harness import Harness
from gustav.testing.fakes import FakeBackend
from gustav.tools.review import check
from gustav.tools.spec import ToolSpec
from gustav.tools.store import ToolStore

pytestmark = pytest.mark.asyncio

ADD = ToolSpec(
    name="add_numbers",
    description="Add two numbers together and return the sum.",
    source="def run(a, b):\n    return a + b",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "the first number"},
            "b": {"type": "number", "description": "the second number"},
        },
        "required": ["a", "b"],
    },
    examples=({"arguments": {"a": 2, "b": 3}, "expect": 5},),
)

#: A tool that works on its examples and fails on other input — which is the realistic
#: case. A tool that failed its own examples could never have been approved.
EXPLODES = ToolSpec(
    name="square_root",
    description="Return the square root of a non-negative number.",
    source=(
        "import math\n"
        "def run(x):\n"
        "    if x < 0:\n"
        "        raise ValueError('x must not be negative')\n"
        "    return math.sqrt(x)"
    ),
    parameters={
        "type": "object",
        "properties": {"x": {"type": "number", "description": "a non-negative number"}},
        "required": ["x"],
    },
    examples=({"arguments": {"x": 9}, "expect": 3.0},),
)


def call(name: str, **arguments) -> str:
    return (
        "Let me work that out.\n<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
    )


class ScriptedBackend(FakeBackend):
    """A fake whose answers come from a list, so a tool round can be scripted."""

    def __init__(self, answers: list[str], **kwargs):
        super().__init__(**kwargs)
        self.answers = list(answers)
        self.index = 0

    def _answer(self, messages):
        reply = self.answers[min(self.index, len(self.answers) - 1)]
        self.index += 1
        return reply


@pytest.fixture
def store(paths, db, repo_root):
    paths.registry.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(repo_root / "modules" / "registry.yaml", paths.registry)
    return ToolStore(db, paths.tools)


@pytest.fixture
def approved(store, db):
    def add(spec: ToolSpec, *, approve: bool = True):
        config = Config().tools
        store.propose(spec, check(spec, config))
        if approve:
            store.approve(spec.name, by="test")
        return store

    return add


@pytest_asyncio.fixture
async def harness_for(paths, db, events, repo_root):
    async def build(answers: list[str], **overrides):
        config = Config.model_validate({"logging": {"console": False}, **overrides})
        hardware = HardwareService(
            config, paths, db, events, detector=simulated("small-12gb"), ladder=load_ladder()
        )
        hardware.ensure_plan()
        backend = ScriptedBackend(answers, base_model="qwen3-4b")
        instance = Harness(
            config=config, paths=paths, hardware=hardware, events=events,
            telemetry=Telemetry(db), db=db, backend=backend,
        )
        await instance.load()
        return instance, backend

    return build


# --- what the model is offered ------------------------------------------
async def test_only_approved_tools_are_offered(harness_for, approved, store):
    approved(ADD, approve=False)
    harness, backend = await harness_for(["no tools for me"])
    assert harness.tool_session() is None, "nothing approved means no tool session"

    store.approve("add_numbers", by="test")
    session = harness.tool_session()
    assert session is not None
    assert [s["function"]["name"] for s in session.schemas()] == ["add_numbers"]


async def test_the_tool_schema_reaches_the_prompt(harness_for, approved):
    approved(ADD)
    harness, backend = await harness_for(["5"])
    await harness.respond(session_id=1, message="what is 2 + 3?", history=[])

    assert backend.offered_tools and backend.offered_tools[0] == ["add_numbers"]
    offered = backend.params[0].tools[0]["function"]
    assert offered["name"] == "add_numbers"
    assert "Add two numbers" in offered["description"]
    assert offered["parameters"]["required"] == ["a", "b"]


async def test_a_model_whose_template_cannot_do_tools_is_not_offered_any(
    harness_for, approved
):
    approved(ADD)
    harness, backend = await harness_for(["5"])
    backend.tools_supported = False
    assert harness.tool_session() is None


async def test_tools_can_be_switched_off(harness_for, approved):
    approved(ADD)
    harness, _backend = await harness_for(["5"], tools={"enabled": False})
    assert harness.tool_session() is None


# --- a round trip -------------------------------------------------------
async def test_a_tool_call_runs_and_its_result_comes_back(harness_for, approved, db):
    approved(ADD)
    harness, backend = await harness_for([call("add_numbers", a=2, b=3), "It is 5."])

    reply = await harness.respond(session_id=7, message="what is 2 + 3?", history=[])
    assert reply.text == "It is 5."
    assert reply.meta["tools"] == ["add_numbers"]
    assert "add_numbers(a=2, b=3) -> ok" in reply.meta["tool_calls"]

    # The second round of generation saw the tool's answer.
    final_prompt = backend.prompts[-1]
    tool_messages = [m for m in final_prompt if m["role"] == "tool"]
    assert tool_messages and tool_messages[0]["content"] == "5"

    logged = db.query_one("SELECT * FROM tool_calls ORDER BY id DESC")
    assert logged["name"] == "add_numbers" and logged["ok"] == 1
    assert json.loads(logged["arguments"]) == {"a": 2, "b": 3}
    assert logged["session_id"] == 7


async def test_the_markup_is_never_shown_to_the_user(harness_for, approved):
    approved(ADD)
    harness, _ = await harness_for([call("add_numbers", a=1, b=1), "The answer is 2."])
    reply = await harness.respond(session_id=1, message="1+1?", history=[])
    assert "<tool_call>" not in reply.text and "</tool_call>" not in reply.text


async def test_a_failing_tool_tells_the_model_rather_than_hiding_it(
    harness_for, approved, db
):
    approved(EXPLODES)
    harness, backend = await harness_for(
        [call("square_root", x=-1), "I could not do that, sorry."]
    )
    reply = await harness.respond(session_id=1, message="try it", history=[])

    tool_messages = [m for m in backend.prompts[-1] if m["role"] == "tool"]
    assert "ValueError" in tool_messages[0]["content"]
    assert reply.text == "I could not do that, sorry."
    assert db.query_one("SELECT ok, error FROM tool_calls")["ok"] == 0


async def test_a_call_to_a_tool_that_does_not_exist_is_answered_not_raised(
    harness_for, approved, db
):
    approved(ADD)
    harness, backend = await harness_for([call("teleport", where="mars"), "I cannot."])
    reply = await harness.respond(session_id=1, message="teleport me", history=[])
    tool_messages = [m for m in backend.prompts[-1] if m["role"] == "tool"]
    assert "no approved tool called 'teleport'" in tool_messages[0]["content"]
    assert reply.text == "I cannot."


async def test_the_call_budget_is_enforced(harness_for, approved, db):
    approved(ADD)
    # A model that only ever asks for the tool again: the loop has to end anyway.
    harness, _ = await harness_for(
        [call("add_numbers", a=1, b=1)], tools={"max_calls_per_turn": 2}
    )
    await harness.respond(session_id=1, message="1+1?", history=[])
    assert db.query_one("SELECT COUNT(*) AS n FROM tool_calls")["n"] == 2


async def test_calls_are_serial_one_per_round(harness_for, approved, db):
    """Generation stops at the closing tag, so a round is one call and its result.

    That is deliberate. Letting the model carry on past its own tool call means letting
    it write down what the tool returned before the tool has run — and it will. The
    cost is that two tools take two rounds; the budget is per answer, so it is paid for.
    """
    approved(ADD)
    harness, backend = await harness_for(
        [call("add_numbers", a=1, b=1), call("add_numbers", a=2, b=2), "2 and 4."]
    )
    reply = await harness.respond(session_id=1, message="both please", history=[])

    assert reply.meta["tools"] == ["add_numbers", "add_numbers"]
    results = [row["result"] for row in db.query("SELECT result FROM tool_calls ORDER BY id")]
    assert results == ["2", "4"]
    assert reply.text == "2 and 4."


async def test_parallel_markup_is_cut_at_the_first_call(harness_for, approved, db):
    """A model that emits two calls in one block gets the first one run, not both."""
    approved(ADD)
    both = (
        '<tool_call>{"name":"add_numbers","arguments":{"a":1,"b":1}}</tool_call>'
        '<tool_call>{"name":"add_numbers","arguments":{"a":9,"b":9}}</tool_call>'
    )
    harness, _ = await harness_for([both, "it is 2."])
    reply = await harness.respond(session_id=1, message="both please", history=[])
    assert reply.meta["tools"] == ["add_numbers"]
    assert db.query_one("SELECT result FROM tool_calls")["result"] == "2"


# --- streaming ----------------------------------------------------------
async def test_streaming_hides_the_call_and_streams_the_answer(harness_for, approved):
    approved(ADD)
    harness, _ = await harness_for([call("add_numbers", a=2, b=3), "It is 5."])
    chunks = [
        chunk.text
        async for chunk in harness.stream(session_id=1, message="2+3?", history=[])
    ]
    streamed = "".join(chunks)
    assert "tool_call" not in streamed
    assert "It is 5." in streamed


async def test_streaming_without_tools_is_unchanged(harness_for, approved):
    harness, _ = await harness_for(["plain answer"])
    chunks = [
        chunk.text
        async for chunk in harness.stream(session_id=1, message="hello", history=[])
    ]
    assert "plain answer" in "".join(chunks)
