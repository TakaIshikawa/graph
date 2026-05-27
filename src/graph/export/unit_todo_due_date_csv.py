"""CSV export for markdown task due-date markers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "task_text", "completed", "due_date", "marker_style"]
_TASK_RE = re.compile(r"^\s*[-*+]\s+\[([ xX])\]\s+(.*)$")
_DUE_PATTERNS = [
    ("due_colon", re.compile(r"\bdue:(\d{4}-\d{2}-\d{2})\b")),
    ("at_due", re.compile(r"@due\((\d{4}-\d{2}-\d{2})\)")),
    ("due_dataview", re.compile(r"\[due::\s*(\d{4}-\d{2}-\d{2})\]")),
]


def export_units_to_todo_due_date_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            task = _TASK_RE.match(line)
            if not task:
                continue
            due = _due(task.group(2))
            if due is None:
                continue
            marker_style, due_date = due
            rows.append({"unit_id": unit_id(unit), "title": title, "line_number": line_number, "task_text": task.group(2).strip(), "completed": "true" if task.group(1).lower() == "x" else "false", "due_date": due_date, "marker_style": marker_style})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _due(text: str) -> tuple[str, str] | None:
    for style, pattern in _DUE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            return style, date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            return None
    return None
