"""CSV export for Markdown checkbox states."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "marker", "state", "text"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TASK_RE = re.compile(r"^\s*(?P<marker>(?:[-+*]|\d+[.)]))\s+\[(?P<box>[ xX\-?])\]\s*(?P<text>.*)$")


def export_units_to_markdown_checkbox_state_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _TASK_RE.match(line)
        if match:
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "title": title,
                    "line_number": line_number,
                    "marker": match.group("marker"),
                    "state": _state(match.group("box")),
                    "text": field_value(match.group("text")),
                }
            )
    return rows


def _state(marker: str) -> str:
    if marker == " ":
        return "open"
    if marker.casefold() == "x":
        return "done"
    if marker == "-":
        return "blocked"
    return "unknown"
