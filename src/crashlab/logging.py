"""Structured logging helpers with optional Rich console and JSON-lines output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOGGERS_CONFIGURED: set[str] = set()


class JsonLinesFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }:
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def _try_rich_handler() -> logging.Handler | None:
    try:
        from rich.logging import RichHandler
    except ImportError:
        return None
    return RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=False,
    )


def configure_logging(
    *,
    level: int = logging.INFO,
    json_log_path: Path | None = None,
    force: bool = False,
) -> None:
    """Configure root crashlab logging once per process."""
    root = logging.getLogger("crashlab")
    if root.handlers and not force:
        return

    if force:
        root.handlers.clear()

    root.setLevel(level)
    root.propagate = False

    console_handler = _try_rich_handler()
    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
    console_handler.setLevel(level)
    root.addHandler(console_handler)

    if json_log_path is not None:
        json_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(json_log_path, encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        file_handler.setLevel(level)
        root.addHandler(file_handler)


def get_logger(name: str, *, level: int | None = None) -> logging.Logger:
    """Return a namespaced logger under ``crashlab``."""
    logger_name = f"crashlab.{name}" if not name.startswith("crashlab") else name

    if logger_name not in _LOGGERS_CONFIGURED:
        configure_logging()
        _LOGGERS_CONFIGURED.add(logger_name)

    logger = logging.getLogger(logger_name)
    if level is not None:
        logger.setLevel(level)
    return logger
