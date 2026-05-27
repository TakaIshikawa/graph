"""CSV export for Markdown checklist task priority markers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "line", "checked", "priority", "marker", "task_text"]
_TASK_RE = re.compile(r"^\s*[-*+]\s+\[(?P<checked>[ xX])\]\s+(?P<text>.*)$")
_TAG_RE = re.compile(r"(?<!\w)#priority/(?P<value>[A-Za-z0-9_-]+)")
_DOUBLE_COLON_RE = re.compile(r"\bpriority::\s*(?P<value>[A-Za-z0-9_-]+)", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[priority:\s*(?P<value>[A-Za-z0-9_-]+)\]", re.IGNORECASE)
_BANG_RE = re.compile(r"^(?P<marker>!{1,3})(?=\s|$)")


def export_units_to_markdown_task_priority_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["priority"]), sort_key(row["task_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        task_match = _TASK_RE.match(line)
        if not task_match:
            continue
        task_text = field_value(task_match.group("text"))
        marker, priority, cleaned = _priority(task_text)
        if not marker:
            continue
        rows.append(
            {
                "unit_id": unit_id(unit),
                "line": line_number,
                "checked": "true" if task_match.group("checked").casefold() == "x" else "false",
                "priority": priority,
                "marker": marker,
                "task_text": cleaned,
            }
        )
    return rows


def _priority(text: str) -> tuple[str, str, str]:
    for pattern in (_TAG_RE, _DOUBLE_COLON_RE, _BRACKET_RE):
        match = pattern.search(text)
        if match:
            marker = match.group(0)
            return marker, _normalize_priority(match.group("value")), _clean(text[: match.start()] + text[match.end() :])
    match = _BANG_RE.match(text)
    if match:
        marker = match.group("marker")
        priority = {"!": "low", "!!": "medium", "!!!": "high"}[marker]
        return marker, priority, _clean(text[match.end() :])
    return "", "", text


def _normalize_priority(value: str) -> str:
    return field_value(value).casefold().replace("_", "-")


def _clean(value: str) -> str:
    return field_value(value.strip(" -:;"))
