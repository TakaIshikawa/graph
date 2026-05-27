"""CSV export for Markdown checkbox task status by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, inline_text, render_csv, unit_id, write_csv

_FIELDNAMES = [
    "unit_id",
    "task_count",
    "completed_count",
    "open_count",
    "canceled_count",
    "completion_ratio",
    "first_open_task",
]
_TASK_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\[([ xX\-~])\]\s*(.*)$")


def export_units_to_checkbox_task_status_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one checkbox task status row per input unit."""
    unit_list = list(units)
    rows = [_unit_row(unit) for unit in unit_list]
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text

    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _unit_row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    counts = {"completed": 0, "open": 0, "canceled": 0}
    first_open_task = ""
    for marker, task_text in _tasks(_content(unit)):
        if marker in {"x", "X"}:
            counts["completed"] += 1
        elif marker == " ":
            counts["open"] += 1
            if not first_open_task:
                first_open_task = inline_text(task_text)
        elif marker in {"-", "~"}:
            counts["canceled"] += 1

    task_count = sum(counts.values())
    ratio = "" if task_count == 0 else f"{counts['completed'] / task_count:.2f}"
    return {
        "unit_id": unit_id(unit),
        "task_count": task_count,
        "completed_count": counts["completed"],
        "open_count": counts["open"],
        "canceled_count": counts["canceled"],
        "completion_ratio": ratio,
        "first_open_task": first_open_task,
    }


def _tasks(content: str) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for line in content.splitlines():
        match = _TASK_RE.match(line)
        if match:
            tasks.append((match.group(1), match.group(2)))
    return tasks


def _content(unit: Mapping[str, Any] | object) -> str:
    value = get(unit, "content")
    return "" if value is None else str(value)
