"""Logging setup: one rotating file plus optional console output."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path

from gustav.config.schema import LoggingConfig

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(config: LoggingConfig, log_file: Path, *, force: bool = False) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("gustav")
    if _CONFIGURED and not force:
        return root
    root.handlers.clear()
    root.setLevel(getattr(logging, config.level))
    root.propagate = False

    plain = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-22s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    formatter: logging.Formatter = JsonFormatter() if config.json_lines else plain

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=config.max_bytes, backupCount=config.backups, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if config.console:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(plain)
        root.addHandler(console)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"gustav.{name}" if not name.startswith("gustav") else name)
