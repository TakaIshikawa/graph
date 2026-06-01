"""CSV export for Markdown task-list due-date markers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "task_text", "due_date", "completed"]
_TASK_RE = re.compile(r"^\s*[-*+]\s+\[(?P<checked>[ xX])\]\s+(?P<text>.*)$")
_DUE_PATTERNS = [
    re.compile(r"\bdue:\s*(?P<date>\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
    re.compile(r"📅\s*(?P<date>\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\[due::\s*(?P<date>\d{4}-\d{2}-\d{2})\]", re.IGNORECASE),
]


def export_units_to_markdown_todo_due_date_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["task_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        task = _TASK_RE.match(line)
        if not task:
            continue
        due_date = _due_date(task.group("text"))
        if due_date is None:
            continue
        rows.append(
            {
                "unit_id": unit_id(unit),
                "title": title,
                "line_number": line_number,
                "task_text": field_value(task.group("text")),
                "due_date": due_date,
                "completed": "true" if task.group("checked").casefold() == "x" else "false",
            }
        )
    return rows


def _due_date(text: str) -> str | None:
    for pattern in _DUE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            return date.fromisoformat(match.group("date")).isoformat()
        except ValueError:
            return None
    return None
