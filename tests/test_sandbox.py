"""The sandbox, which is what makes a practice result a measurement."""

from __future__ import annotations

import pytest

from gustav.learn.code import extract_python, parses
from gustav.sandbox import Sandbox


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox(timeout_seconds=6, memory_limit_mb=512, workdir=tmp_path)


def test_a_correct_solution_passes_and_a_wrong_one_does_not(sandbox):
    good = sandbox.run_python("def add(a, b):\n    return a + b", tests="assert add(1, 2) == 3")
    assert good.passed and good.detail == "passed"
    bad = sandbox.run_python("def add(a, b):\n    return a - b", tests="assert add(1, 2) == 3")
    assert not bad.passed
    assert "AssertionError" in bad.detail


def test_an_infinite_loop_costs_seconds_not_the_night(sandbox):
    result = sandbox.run_python("while True:\n    pass")
    assert result.timed_out and not result.passed
    assert result.duration_ms < 20_000


def test_an_allocation_storm_hits_the_limit(sandbox):
    result = sandbox.run_python("x = bytearray(4_000_000_000)")
    assert not result.passed
    assert "MemoryError" in result.detail or result.timed_out


def test_the_network_and_subprocesses_are_blocked(sandbox):
    assert "not available" in sandbox.run_python("import socket; socket.socket()").detail
    assert "not available" in sandbox.run_python("import os; os.system('true')").detail


def test_a_syntax_error_is_caught_without_starting_a_process():
    assert not Sandbox().check_imports("def f(:").passed
    assert Sandbox().check_imports("def f():\n    pass").passed


def test_empty_code_is_a_failure_not_a_pass(sandbox):
    result = sandbox.run_python("   ")
    assert not result.passed and "no code" in result.detail


def test_output_is_truncated_rather_than_unbounded(tmp_path):
    sandbox = Sandbox(timeout_seconds=10, max_output_bytes=1024, workdir=tmp_path)
    result = sandbox.run_python("print('x' * 100000)")
    assert len(result.stdout) <= 1024


def test_the_candidate_cannot_raise_its_own_limits(sandbox):
    escape = (
        "import resource\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
        "resource.setrlimit(resource.RLIMIT_AS, (hard * 4, hard * 4))\n"
    )
    assert not sandbox.run_python(escape).passed


def test_setup_runs_before_the_candidate(sandbox):
    result = sandbox.run_python(
        "def double(x):\n    return BASE * x", setup="BASE = 2", tests="assert double(3) == 6"
    )
    assert result.passed


# --- getting code out of an answer -------------------------------------
def test_code_is_extracted_from_a_chatty_answer():
    assert extract_python("Sure!\n```python\ndef f():\n    return 1\n```\nDone.") == (
        "def f():\n    return 1"
    )
    assert extract_python("Here:\ndef f():\n    return 1\nHope that helps!") == (
        "def f():\n    return 1"
    )
    assert extract_python("I would rather not.") == ""
    assert parses("x = 1") and not parses("x ===")
