"""CSV export for Markdown heading outlines in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "heading_count", "max_depth", "top_level_headings", "deepest_heading", "has_outline"]
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def export_units_to_markdown_heading_outline_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    headings = _headings(content)
    max_depth = max((depth for depth, _title in headings), default=0)
    top_level_depth = min((depth for depth, _title in headings), default=0)
    top_level = [title for depth, title in headings if depth == top_level_depth]
    deepest = next((title for depth, title in reversed(headings) if depth == max_depth), "")
    return {
        "unit_id": unit_id(unit),
        "heading_count": len(headings),
        "max_depth": max_depth,
        "top_level_headings": "; ".join(top_level),
        "deepest_heading": deepest,
        "has_outline": "true" if headings else "false",
    }


def _headings(content: str) -> list[tuple[int, str]]:
    in_fence = False
    headings: list[tuple[int, str]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append((len(match.group(1)), field_value(match.group(2))))
    return headings
