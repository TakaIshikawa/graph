"""CSV export helpers for ActivityWatch focus sessions."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "session_start",
    "session_end",
    "duration_seconds",
    "app",
    "domain",
    "title",
    "unit_count",
]


def export_units_to_activitywatch_focus_sessions_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_duration_seconds: int = 300,
) -> str | dict[str, Any]:
    """Return or write deterministic ActivityWatch focus-session CSV rows."""
    rows = _session_rows(list(units), min_duration_seconds=min_duration_seconds)
    text = _render_csv(rows)
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "rows_written": len(rows)}


def _session_rows(units: list[KnowledgeUnit], *, min_duration_seconds: int) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in sorted((_activity_item(unit) for unit in units), key=lambda item: (item["start"], item["source_id"])):
        if current is None or not _can_merge(current, item):
            if current is not None:
                sessions.append(current)
            current = {
                "start": item["start"],
                "end": item["end"],
                "app": item["app"],
                "domain": item["domain"],
                "title": item["title"],
                "unit_count": 1,
            }
            continue
        current["end"] = max(current["end"], item["end"])
        current["unit_count"] += 1
    if current is not None:
        sessions.append(current)

    rows = []
    for session in sessions:
        duration = int((session["end"] - session["start"]).total_seconds())
        if duration < min_duration_seconds:
            continue
        rows.append(
            {
                "session_start": session["start"].isoformat(),
                "session_end": session["end"].isoformat(),
                "duration_seconds": duration,
                "app": session["app"],
                "domain": session["domain"],
                "title": session["title"],
                "unit_count": session["unit_count"],
            }
        )
    return rows


def _activity_item(unit: KnowledgeUnit) -> dict[str, Any]:
    start = _parse_datetime(unit.metadata.get("timestamp")) or unit.created_at
    duration = _parse_float(unit.metadata.get("duration")) or 0.0
    return {
        "source_id": unit.source_id,
        "start": start,
        "end": start + timedelta(seconds=duration),
        "app": _text(unit.metadata.get("app")),
        "domain": _text(unit.metadata.get("domain")),
        "title": _text(unit.metadata.get("title") or unit.title),
    }


def _can_merge(session: dict[str, Any], item: dict[str, Any]) -> bool:
    if (session["app"], session["domain"], session["title"]) != (item["app"], item["domain"], item["title"]):
        return False
    gap = (item["start"] - session["end"]).total_seconds()
    return gap <= 5


def _render_csv(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())
