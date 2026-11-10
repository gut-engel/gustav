"""Loading and merging configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from gustav.config.schema import Config
from gustav.paths import Paths

ENV_CONFIG = "GUSTAV_CONFIG"
ENV_DATA_DIR = "GUSTAV_DATA_DIR"


class ConfigError(RuntimeError):
    """Raised with a human-readable explanation of what is wrong in config.yaml."""


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def find_project_root(start: Path | None = None) -> Path:
    """Walk up looking for a config.yaml / pyproject.toml marker."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config.yaml").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return here


def load_config(
    config_path: Path | str | None = None,
    *,
    root: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[Config, Paths]:
    """Load config.yaml (+ the active profile) and derive the path layout.

    Precedence, lowest first: schema defaults, ``config.yaml``, the profile named by
    ``hardware.profile``, explicit ``overrides``, then ``GUSTAV_DATA_DIR``.
    """
    if config_path is None:
        config_path = os.environ.get(ENV_CONFIG)
    project_root = Path(root).resolve() if root else None

    if config_path:
        cfg_file = Path(config_path).expanduser().resolve()
        project_root = project_root or cfg_file.parent
    else:
        project_root = project_root or find_project_root()
        cfg_file = project_root / "config.yaml"

    data = _read_yaml(cfg_file)

    # Overrides are merged first so that they can *select* the profile, and again at
    # the end so that they still win over whatever the profile sets.
    selection = deep_merge(data, overrides) if overrides else data
    profile_name = (selection.get("hardware") or {}).get("profile")
    # A profile file is a hardware section: knobs plus an optional pinned `plan`.
    if profile_name and profile_name != "auto":
        profile_file = project_root / "profiles" / f"{profile_name}.yaml"
        if not profile_file.exists():
            raise ConfigError(
                f"profile {profile_name!r} referenced by {cfg_file} does not exist "
                f"(looked for {profile_file})"
            )
        data = deep_merge(data, {"hardware": _read_yaml(profile_file)})

    if overrides:
        data = deep_merge(data, overrides)

    env_data_dir = os.environ.get(ENV_DATA_DIR)
    if env_data_dir:
        data = deep_merge(data, {"instance": {"data_dir": env_data_dir}})

    try:
        config = Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {cfg_file}:\n{_explain(exc)}") from exc

    paths = Paths.create(project_root, config.instance.data_dir)
    return config, paths


def _explain(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
