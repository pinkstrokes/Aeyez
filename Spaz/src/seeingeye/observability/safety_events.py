"""Safety event logging and lightweight daily summaries."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_preview(text: str, limit: int = 600) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit]


def log_safety_event(
    *,
    log_path: str | Path,
    question: str,
    answer: str,
    sir: str,
    outer_iters_used: int,
    total_tokens: int,
    media_type: str,
) -> dict[str, Any]:
    """Append a compact, image-free safety event to JSONL."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    event = {
        "timestamp_utc": now.isoformat(),
        "date_utc": now.date().isoformat(),
        "question": _safe_preview(question, 240),
        "answer_preview": _safe_preview(answer, 800),
        "sir_preview": _safe_preview(sir, 800),
        "media_type": media_type,
        "outer_iters_used": outer_iters_used,
        "total_tokens": total_tokens,
        "has_safety_report": "SAFETY REPORT" in (answer or ""),
        "has_route_map": "ego_centered_route_map" in (sir or "")
        or "ROUTE MEMORY BLOCK" in (sir or ""),
        "has_mechanics_scan": "MECHANICS HAZARD SCAN" in (sir or "")
        or "mechanical_scene_model" in (sir or ""),
        "has_vstar_synthesis": "V* TOOL COLLABORATION SYNTHESIS" in (sir or ""),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def summarize_daily_events(
    *,
    log_path: str | Path,
    date_utc: str | None = None,
) -> dict[str, Any]:
    """Summarize logged safety events for a UTC date."""
    path = Path(log_path)
    if date_utc is None:
        date_utc = datetime.now(timezone.utc).date().isoformat()
    events: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("date_utc") == date_utc:
                events.append(item)
    flags = Counter()
    for event in events:
        for key in ("has_safety_report", "has_route_map", "has_mechanics_scan", "has_vstar_synthesis"):
            if event.get(key):
                flags[key] += 1
    return {
        "date_utc": date_utc,
        "total_events": len(events),
        "tooling_flags": dict(flags),
        "recent_events": events[-10:],
    }
