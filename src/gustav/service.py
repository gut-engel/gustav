"""Running the daemon as a service.

Deliberately thin, and deliberately a *user* unit: gustav runs as you, with your GPU
access, your files and your home directory. A system unit would need a service account
with access to the GPU and to a project directory it does not own, which is a worse
trade for a personal assistant.

It is not a scheduler. The unit starts one always-on process and restarts it if it
dies; the daemon decides when to sleep by itself, and a timer here would break the one
rule the design is built on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

UNIT_NAME = "gustav.service"
TEMPLATE = Path(__file__).resolve().parents[2] / "packaging" / UNIT_NAME


@dataclass
class ServicePlan:
    unit_path: Path
    workdir: Path
    python: str
    exists: bool
    body: str

    def describe(self) -> list[str]:
        return [
            f"unit file   {self.unit_path}" + ("  (exists)" if self.exists else ""),
            f"working dir {self.workdir}",
            f"python      {self.python}",
        ]


def unit_directory() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def template_text() -> str:
    """The shipped unit, whether running from a checkout or an installed wheel."""
    if TEMPLATE.exists():
        return TEMPLATE.read_text(encoding="utf-8")
    from gustav.data import packaged

    try:
        return packaged(UNIT_NAME).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise RuntimeError(
            f"the {UNIT_NAME} template is missing from this installation"
        ) from exc


def plan(root: Path | str, *, python: str | None = None) -> ServicePlan:
    workdir = Path(root).resolve()
    interpreter = python or sys.executable
    body = (
        template_text()
        .replace("%WORKDIR%", str(workdir))
        .replace("%PYTHON%", interpreter)
    )
    path = unit_directory() / UNIT_NAME
    return ServicePlan(
        unit_path=path,
        workdir=workdir,
        python=interpreter,
        exists=path.exists(),
        body=body,
    )


def install(root: Path | str, *, python: str | None = None, force: bool = False) -> ServicePlan:
    """Write the unit. Refuses to overwrite an existing one unless asked."""
    prepared = plan(root, python=python)
    if prepared.exists and not force:
        raise FileExistsError(
            f"{prepared.unit_path} already exists; pass --force to replace it"
        )
    prepared.unit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.unit_path.with_name(f".{UNIT_NAME}.part")
    temporary.write_text(prepared.body, encoding="utf-8")
    temporary.replace(prepared.unit_path)
    return prepared


def uninstall() -> Path | None:
    path = unit_directory() / UNIT_NAME
    if not path.exists():
        return None
    path.unlink()
    return path


def systemctl(*args: str) -> tuple[int, str]:
    """Run ``systemctl --user``, or explain why it could not."""
    if shutil.which("systemctl") is None:
        return 127, "systemctl is not on PATH (this machine may not use systemd)"
    try:
        done = subprocess.run(
            ["systemctl", "--user", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout or done.stderr).strip()


def status() -> dict[str, object]:
    path = unit_directory() / UNIT_NAME
    code, text = systemctl("is-active", UNIT_NAME)
    enabled_code, enabled_text = systemctl("is-enabled", UNIT_NAME)
    return {
        "unit": str(path),
        "installed": path.exists(),
        "active": text if code != 127 else "unknown",
        "enabled": enabled_text if enabled_code != 127 else "unknown",
        "systemctl": code != 127,
    }
