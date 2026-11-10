"""Atomic filesystem moves.

Everything here exists so that a crash at any instant leaves either the old state or the
new one. The rule is always the same: build the replacement somewhere else, then make it
live with a single ``rename``, which the kernel either does or does not do.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def remove(path: Path) -> None:
    """Delete a file, symlink or directory if it is there."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def atomic_write(path: Path, data: str | bytes, *, mode: int | None = None) -> Path:
    """Write via a temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    if isinstance(data, str):
        temporary.write_text(data, encoding="utf-8")
    else:
        temporary.write_bytes(data)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)
    return path


def atomic_symlink(link: Path, target: str | Path) -> Path:
    """Point ``link`` at ``target``, with no instant where the link is missing."""
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.swap")
    remove(temporary)
    temporary.symlink_to(target)
    os.replace(temporary, link)
    return link


def promote(staging: Path, target: Path) -> None:
    """Make ``staging`` take the place of ``target``.

    The old target is renamed aside first and removed only once the new one is live, so
    a failure half-way puts the original back rather than leaving nothing.
    """
    if not staging.exists() and not staging.is_symlink():
        raise FileNotFoundError(f"nothing staged at {staging}")
    target.parent.mkdir(parents=True, exist_ok=True)
    retired = target.with_name(f".{target.name}.retired")
    remove(retired)
    had_target = target.exists() or target.is_symlink()
    try:
        if had_target:
            os.replace(target, retired)
        os.replace(staging, target)
    except OSError:
        if had_target and not (target.exists() or target.is_symlink()):
            os.replace(retired, target)
        raise
    finally:
        remove(retired)


@contextmanager
def staged(tmp_root: Path, name: str) -> Iterator[Path]:
    """A scratch directory that is cleaned up whether or not the block succeeds."""
    tmp_root.mkdir(parents=True, exist_ok=True)
    staging = tmp_root / f"{name}.staging"
    remove(staging)
    staging.mkdir(parents=True)
    try:
        yield staging
    finally:
        remove(staging)
