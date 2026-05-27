"""CSV export for Markdown heading hierarchy within units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "level", "heading_text", "line_number", "parent_path", "skipped_level"]
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def export_units_to_heading_hierarchy_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    stack: list[tuple[int, str]] = []
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        text = field_value(match.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_path = " > ".join(item[1] for item in stack)
        expected = stack[-1][0] + 1 if stack else 1
        rows.append(
            {
                "unit_id": unit_id(unit),
                "title": title,
                "level": level,
                "heading_text": text,
                "line_number": line_number,
                "parent_path": parent_path,
                "skipped_level": "true" if level > expected else "false",
            }
        )
        stack.append((level, text))
    return rows
