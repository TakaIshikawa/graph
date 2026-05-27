"""CSV export for Markdown table-of-contents markers by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "toc_marker_count", "html_toc_comment_count", "bracket_toc_marker_count", "first_marker_line"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HTML_RE = re.compile(r"<!--\s*toc\s*-->", re.IGNORECASE)
_BRACKET_RE = re.compile(r"(?<!\[)\[{1,2}toc\]{1,2}(?!\])", re.IGNORECASE)


def export_units_to_toc_marker_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    html_count = 0
    bracket_count = 0
    first_line = 0
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line_html = len(_HTML_RE.findall(line))
        line_bracket = len(_BRACKET_RE.findall(line))
        if first_line == 0 and line_html + line_bracket:
            first_line = line_number
        html_count += line_html
        bracket_count += line_bracket
    return {
        "unit_id": unit_id(unit),
        "toc_marker_count": html_count + bracket_count,
        "html_toc_comment_count": html_count,
        "bracket_toc_marker_count": bracket_count,
        "first_marker_line": first_line,
    }
