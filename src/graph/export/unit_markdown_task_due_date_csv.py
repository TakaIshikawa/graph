"""CSV export for Markdown task due date markers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "checked", "due_date", "task_text"]
_TASK_RE = re.compile(r"^\s*[-*+]\s+\[(?P<mark>[ xX])\]\s+(?P<text>.*)$")
_DUE_PATTERNS = [
    re.compile(r"(?:📅|due:)\s*(?P<date>\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    re.compile(r"\[due::\s*(?P<date>\d{4}-\d{2}-\d{2})\]", re.IGNORECASE),
]


def export_units_to_markdown_task_due_date_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["due_date"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int | bool]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        task = _TASK_RE.match(line)
        if not task:
            continue
        text = task.group("text")
        due_date = ""
        cleaned = text
        for pattern in _DUE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            due_date = _normalize_date(match.group("date"))
            cleaned = (text[: match.start()] + text[match.end() :]).strip()
            break
        if not due_date:
            continue
        rows.append({"unit_id": unit_id(unit), "title": title, "line_number": line_number, "checked": str(task.group("mark").casefold() == "x").lower(), "due_date": due_date, "task_text": field_value(cleaned)})
    return rows


def _normalize_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return value
