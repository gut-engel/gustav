"""Structural tests: the guard must be the *only* way weights change.

Section 9 is an invariant, not a convention, so it is checked by reading the source
rather than trusting everyone to remember. These tests fail when a new code path writes
to adapters without going through ``WeightChangeGuard``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "gustav"

#: Calls that change which weights are live. Each must sit inside a guard block.
WEIGHT_CHANGING_CALLS = {"activate_adapter", "rollback_adapter", "approve_base_change"}

#: Raw filesystem mutations, matched on the receiver as well as the name so that
#: `datetime.replace` and `str.replace` are not mistaken for `os.replace`.
MUTATING_CALLS = {
    ("shutil", "rmtree"),
    ("os", "replace"),
    ("os", "rename"),
    ("os", "removedirs"),
    (None, "symlink_to"),   # any receiver: this only ever means a symlink
}
ALLOWED_MUTATORS = {
    "backup/atomic.py",
    "backup/restore.py",
    "backup/snapshot.py",
    "backup/manifest.py",
    "inference/adapters.py",
    "core/recovery.py",       # clears staging directories, which are by definition dead
    "inference/models.py",    # downloaded weights, before anything is live
    "testing/fakes.py",
}


def modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def relative(path: Path) -> str:
    return str(path.relative_to(SRC))


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_guard_change(node: ast.expr) -> bool:
    """Does this expression call ``*.guard.change(...)``?"""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "change" and (
        isinstance(func.value, ast.Attribute) and func.value.attr == "guard"
    )


def _guarded_regions(tree: ast.Module) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            if any(_is_guard_change(item.context_expr) for item in node.items):
                regions.append((node.lineno, node.end_lineno or node.lineno))
    return regions


def test_the_daemon_only_changes_adapters_inside_the_guard():
    """Every activate/rollback call must be lexically inside `with self.guard.change(...)`."""
    path = SRC / "daemon.py"
    tree = parse(path)
    regions = _guarded_regions(tree)
    assert regions, "daemon.py does not use the guard at all"

    unguarded: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in WEIGHT_CHANGING_CALLS:
            continue
        # Calling it on the daemon itself (self.activate_adapter) is the guarded wrapper.
        if isinstance(func.value, ast.Name) and func.value.id == "self":
            continue
        if not any(start <= node.lineno <= end for start, end in regions):
            unguarded.append(f"{relative(path)}:{node.lineno} {func.attr}")
    assert not unguarded, (
        "these weight changes are not inside a guard block: " + ", ".join(unguarded)
    )


def _receiver(node: ast.Attribute) -> str | None:
    value = node.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def test_only_the_store_and_the_backup_package_mutate_the_filesystem():
    offenders: list[str] = []
    for path in modules():
        name = relative(path)
        if name in ALLOWED_MUTATORS:
            continue
        for node in ast.walk(parse(path)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            receiver, attribute = _receiver(node.func), node.func.attr
            if (receiver, attribute) in MUTATING_CALLS or (None, attribute) in MUTATING_CALLS:
                offenders.append(f"{name}:{node.lineno} {receiver}.{attribute}()")
    assert not offenders, (
        "filesystem mutations outside the adapter store and the backup package:\n  "
        + "\n  ".join(offenders)
        + "\nIf this is deliberate, add the module to ALLOWED_MUTATORS and say why."
    )


def test_the_adapter_store_is_the_only_writer_of_the_adapters_tree():
    """Nothing else may construct paths under `paths.adapters` and write to them."""
    offenders: list[str] = []
    allowed = {"inference/adapters.py", "inference/harness.py", "backup/restore.py",
               "backup/snapshot.py", "testing/fakes.py"}
    for path in modules():
        name = relative(path)
        if name in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        for number, line in enumerate(source.splitlines(), start=1):
            if "paths.adapters" in line and any(
                token in line for token in ("mkdir", "write_text", "write_bytes", "open(")
            ):
                offenders.append(f"{name}:{number}")
    assert not offenders, "direct writes under adapters/: " + ", ".join(offenders)


def test_the_guard_refuses_rather_than_continuing_without_a_backup():
    """The refusal path must raise, not log-and-carry-on."""
    source = (SRC / "backup" / "guard.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    raises = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "BackupRequired"
    ]
    assert raises, "guard.py never raises BackupRequired"


def test_every_crash_rollback_step_has_a_handler_registered():
    """A step with the `rollback` crash policy needs something to roll it back."""
    from gustav.config.schema import Config
    from gustav.core.journal import CrashPolicy
    from gustav.sleep.pipeline import default_pipeline

    needs_rollback = {
        step.spec.name
        for step in default_pipeline(Config()).steps
        if step.spec.on_crash is CrashPolicy.ROLLBACK
    }
    source = (SRC / "daemon.py").read_text(encoding="utf-8")
    for name in needs_rollback:
        assert f'"{name}"' in source, (
            f"step {name!r} rolls back on crash but the daemon registers no handler for it"
        )
