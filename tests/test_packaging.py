"""Packaging: a wheel in an empty directory has to become a working project."""

from __future__ import annotations

import subprocess
import sys

import pytest

from gustav.data import DEFAULTS, install_defaults, packaged


def test_every_default_file_ships_with_the_package():
    for name in DEFAULTS:
        path = packaged(name)
        assert path.exists() and path.stat().st_size > 0, name


def test_the_shipped_registry_and_sources_are_valid():
    from gustav.fetcher.sources import load_sources
    from gustav.inference.registry import load_registry

    registry = load_registry(packaged("registry.yaml"))
    assert len(registry) >= 3 and "general" in registry
    sources = load_sources(packaged("sources.yaml"))
    assert sources.entries and not sources.warnings


def test_the_shipped_config_is_loadable_and_complete(tmp_path):
    """An unknown key in it would make every fresh install fail at startup."""
    import shutil

    from gustav.config.load import load_config

    shutil.copy(packaged("config.yaml"), tmp_path / "config.yaml")
    config, paths = load_config(tmp_path / "config.yaml")
    assert config.instance.name
    assert paths.root == tmp_path


def test_init_writes_the_defaults_once_and_never_overwrites(tmp_path):
    first = install_defaults(tmp_path)
    assert {path.name for path in first.created} == {
        "registry.yaml", "sources.yaml", "config.yaml"
    }
    (tmp_path / "modules" / "registry.yaml").write_text("modules: []\n", encoding="utf-8")

    second = install_defaults(tmp_path)
    assert second.created == []
    assert (tmp_path / "modules" / "registry.yaml").read_text(encoding="utf-8") == (
        "modules: []\n"
    ), "your edits survive"


def test_a_dry_run_writes_nothing(tmp_path):
    result = install_defaults(tmp_path, dry_run=True)
    assert result.created and not any(path.exists() for path in result.created)


def test_only_some_defaults_can_be_asked_for(tmp_path):
    result = install_defaults(tmp_path, only=("registry.yaml",))
    assert [path.name for path in result.created] == ["registry.yaml"]
    assert not (tmp_path / "config.yaml").exists()


# --- the service unit ---------------------------------------------------
def test_the_unit_is_a_user_unit_with_no_timer():
    from gustav import service

    body = service.template_text()
    assert "[Install]" in body and "WantedBy=default.target" in body
    assert "OnCalendar" not in body and "Timer" not in body, (
        "the daemon schedules itself; a timer here would break that"
    )
    assert "Restart=on-failure" in body
    assert "TimeoutStopSec" in body, "it needs time to checkpoint the night"


def test_the_plan_fills_in_this_project_and_interpreter(tmp_path):
    from gustav import service

    plan = service.plan(tmp_path, python="/usr/bin/python3")
    assert "%WORKDIR%" not in plan.body and "%PYTHON%" not in plan.body
    assert str(tmp_path) in plan.body
    assert "/usr/bin/python3" in plan.body
    assert plan.unit_path.name == "gustav.service"


def test_installing_refuses_to_clobber_an_existing_unit(tmp_path, monkeypatch):
    from gustav import service

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    written = service.install(tmp_path)
    assert written.unit_path.exists()
    with pytest.raises(FileExistsError, match="--force"):
        service.install(tmp_path)
    service.install(tmp_path, force=True)
    assert service.uninstall() == written.unit_path
    assert service.uninstall() is None


def test_status_is_honest_on_a_machine_without_systemd(monkeypatch, tmp_path):
    from gustav import service

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(service.shutil, "which", lambda _name: None)
    state = service.status()
    assert state["systemctl"] is False
    assert state["active"] == "unknown"


# --- the CLI, with no daemon -------------------------------------------
def run_cli(cwd, *args):
    import os

    env = {**os.environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[1] / "src")}
    return subprocess.run(
        [sys.executable, "-m", "gustav.cli", *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
    )


@pytest.mark.slow
def test_ai_init_needs_no_daemon(tmp_path):
    result = run_cli(tmp_path, "init")
    assert result.returncode == 0, result.stderr
    assert "wrote config.yaml" in result.stdout
    assert (tmp_path / "modules" / "registry.yaml").exists()
    assert (tmp_path / "var").is_dir()

    again = run_cli(tmp_path, "init")
    assert "already a gustav project" in again.stdout


@pytest.mark.slow
def test_ai_service_dry_run_prints_the_unit(tmp_path):
    result = run_cli(tmp_path, "service", "install", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "ExecStart=" in result.stdout
    assert str(tmp_path) in result.stdout
