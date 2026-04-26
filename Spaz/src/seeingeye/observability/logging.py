"""Loguru JSONL sink with content filters for the SeeingEye framework.

Prevents image data (base64-encoded) from leaking into logs or external
trace services (Pitfall 15), and ensures LangSmith tracing is off by
default to avoid unexpected network calls in the HPC environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


def _content_filter(record: dict) -> bool:
    """Drop log records containing base64 image data or exceeding 4 KB.

    Prevents image-data leaks into logs or external trace services
    (per Pitfall 15 in PITFALLS.md).

    Args:
        record: loguru record dict — ``record["message"]`` holds the log text.

    Returns:
        True to keep the record, False to drop it.
    """
    msg = record.get("message", "")
    if "base64," in msg:
        return False
    if len(msg) > 4096:
        return False
    return True


def configure_logging(
    log_dir: str | Path = "logs",
    jsonl: bool = True,
    console_level: str = "INFO",
) -> None:
    """Configure loguru with a JSONL sink and content filters.

    Sets ``LANGSMITH_TRACING=false`` and ``LANGCHAIN_TRACING_V2=false``
    by default to prevent unexpected network calls in HPC environments.
    Existing env-var values are **not** overridden (``setdefault``).

    Args:
        log_dir: Directory for the JSONL log file.
        jsonl: If True, add a JSONL file sink with ``serialize=True``.
        console_level: Minimum level for the stderr console sink.
    """
    # Ensure tracing is off by default (won't override if user already set)
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

    logger.remove()
    logger.add(sys.stderr, level=console_level, filter=_content_filter)

    if jsonl:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path / "seeingeye.jsonl"),
            serialize=True,
            level="DEBUG",
            filter=_content_filter,
            rotation="50 MB",
        )
