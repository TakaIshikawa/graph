"""CSV export for Markdown list structure by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, unit_id, write_csv

_FIELDNAMES = ["unit_id", "unordered_items", "ordered_items", "task_items", "max_indent_level", "list_line_count"]
_LIST_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>(?:[-+*]|\d+[.)]))\s+(?P<rest>.*)$")
_TASK_RE = re.compile(r"^\[[^\]\n]*\]\s+")


def export_units_to_markdown_list_structure_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [_row(unit) for unit in unit_list]
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, int | str]:
    unordered = ordered = tasks = list_lines = max_depth = 0
    for line in str(get(unit, "content") or "").splitlines():
        match = _LIST_RE.match(line)
        if not match:
            continue
        list_lines += 1
        marker = match.group("marker")
        rest = match.group("rest")
        depth = len(match.group("indent").replace("\t", "    ")) // 2
        max_depth = max(max_depth, depth)
        if marker[0].isdigit():
            ordered += 1
        else:
            unordered += 1
        if _TASK_RE.match(rest):
            tasks += 1
    return {
        "unit_id": unit_id(unit),
        "unordered_items": unordered,
        "ordered_items": ordered,
        "task_items": tasks,
        "max_indent_level": max_depth,
        "list_line_count": list_lines,
    }
