"""The files a project needs, shipped inside the package.

``modules/registry.yaml`` and ``docs/sources.yaml`` are *yours*: they decide which
modules exist and which documentation may ever be fetched, and they are meant to be
edited. But an installed wheel starting in an empty directory has neither, and a system
with no modules and no sources is not a useful default — so the defaults travel with the
package and are copied into the project on first run, once, and never again.

Copied, not read in place, for two reasons: a file under `site-packages` is not somewhere
anyone would think to edit, and an upgrade must not silently change the registry a
trained adapter belongs to.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

#: packaged name -> where it belongs in a project.
DEFAULTS: dict[str, str] = {
    "registry.yaml": "modules/registry.yaml",
    "sources.yaml": "docs/sources.yaml",
    "config.yaml": "config.yaml",
}

#: Also shipped, but not copied into a project: the systemd user unit, which
#: `ai service install` writes to ~/.config/systemd/user instead.
EXTRA = ("gustav.service",)


def packaged(name: str) -> Path:
    """Where the shipped copy of ``name`` lives."""
    with resources.as_file(resources.files("gustav.data") / name) as path:
        return Path(path)


@dataclass
class Installed:
    created: list[Path]
    kept: list[Path]

    def describe(self) -> list[str]:
        return [f"created {path}" for path in self.created] + [
            f"kept your {path}" for path in self.kept
        ]


def install_defaults(
    root: Path | str, *, only: tuple[str, ...] = (), dry_run: bool = False
) -> Installed:
    """Put the default files into a project, without ever overwriting one."""
    root = Path(root)
    created: list[Path] = []
    kept: list[Path] = []
    for name, relative in DEFAULTS.items():
        if only and name not in only:
            continue
        target = root / relative
        if target.exists():
            kept.append(target)
            continue
        created.append(target)
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packaged(name), target)
    return Installed(created=created, kept=kept)
