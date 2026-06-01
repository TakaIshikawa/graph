"""CSV export for Markdown definition-list entries in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "term", "definition", "line_number", "source_url"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^\s*:\s+(.*\S)\s*$")


def export_units_to_markdown_definition_list_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["term"]), sort_key(row["definition"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source_url = field_value(get(unit, "source_url") or data.get("source_url") or get(unit, "url") or data.get("url"))
    rows: list[dict[str, str | int]] = []
    term = ""
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _DEF_RE.match(line)
        if match and term:
            rows.append({"unit_id": uid, "title": title, "term": term, "definition": field_value(match.group(1)), "line_number": line_number, "source_url": source_url})
        elif line.strip():
            term = field_value(line)
        else:
            term = ""
    return rows
