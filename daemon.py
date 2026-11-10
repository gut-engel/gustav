#!/usr/bin/env python3
"""Entry point: ``python daemon.py``.

One always-on process runs everything — chat, scheduler, sleep pipeline, supervision.
No cron, no systemd timer, no external scheduler. See README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a checkout without installing.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gustav.daemon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
