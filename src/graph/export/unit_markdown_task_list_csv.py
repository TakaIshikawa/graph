"""CSV export for GitHub-style Markdown task list items."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "checked", "marker", "nesting_depth", "task_text", "parent_heading"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.*?)(?:\s+#+\s*)?$")
_TASK_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-+*])\s+\[(?P<box>[ xX])\]\s+(?P<text>.*)$")
_PATH_KEYS = ("path", "source_path", "file_path", "filename", "source_url")
_SOURCE_KEYS = ("source", "source_name", "source_id")


def export_units_to_markdown_task_list_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["task_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    rows: list[dict[str, str | int]] = []
    in_fence = False
    parent_heading = ""
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            parent_heading = field_value(heading.group("text"))
            continue
        match = _TASK_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "unit_id": unit_id(unit),
                "title": field_value(get(unit, "title") or data.get("title")),
                "source_path": _first_value(unit, data, _PATH_KEYS),
                "source": _first_value(unit, data, _SOURCE_KEYS),
                "line_number": line_number,
                "checked": str(match.group("box").casefold() == "x").lower(),
                "marker": match.group("marker"),
                "nesting_depth": _depth(match.group("indent")),
                "task_text": field_value(match.group("text")),
                "parent_heading": parent_heading,
            }
        )
    return rows


def _depth(indent: str) -> int:
    return sum(4 if char == "\t" else 1 for char in indent) // 2


def _first_value(unit: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = field_value(get(unit, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
