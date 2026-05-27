"""CSV export for explicit Markdown heading anchors."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "level", "heading_text", "anchor_id", "line_number", "context"]
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s+\{#([A-Za-z0-9_.:-]+)\}\s*#*\s*$")


def export_units_to_markdown_heading_anchor_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["anchor_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if match := _HEADING_RE.match(line):
            rows.append({"unit_id": uid, "title": title, "level": len(match.group(1)), "heading_text": field_value(match.group(2)), "anchor_id": match.group(3), "line_number": line_number, "context": field_value(line)})
    return rows
