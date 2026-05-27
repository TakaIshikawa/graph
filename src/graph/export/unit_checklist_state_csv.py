"""CSV export for Markdown checklist item states by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "state_marker", "state", "item_text", "line_number", "indentation_depth"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TASK_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-+*]|\d+[.)])\s+\[(?P<marker>[^\]\n]*)\]\s*(?P<text>.*)$")


def export_units_to_checklist_state_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _TASK_RE.match(line)
        if match:
            marker = match.group("marker")
            rows.append({
                "unit_id": unit_id(unit),
                "title": field_value(get(unit, "title")),
                "state_marker": marker,
                "state": _state(marker),
                "item_text": field_value(match.group("text")),
                "line_number": line_number,
                "indentation_depth": len(match.group("indent").replace("\t", "    ")) // 2,
            })
    return rows


def _state(marker: str) -> str:
    if marker == " " or marker == "":
        return "open"
    if marker.casefold() == "x":
        return "done"
    return "custom"
