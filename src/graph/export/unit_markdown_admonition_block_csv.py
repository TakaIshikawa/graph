"""CSV export for Markdown admonition block starts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "admonition_type", "marker_style", "line_number", "title_text"]
_CALLOUT_RE = re.compile(r"^\s{0,3}>\s*\[!([A-Za-z][\w-]*)\][+-]?\s*(.*)$")
_COLON_RE = re.compile(r"^\s{0,3}:::+\s*([A-Za-z][\w-]*)(?:\s+(.+))?\s*$")


def export_units_to_markdown_admonition_block_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["admonition_type"])))
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
        if match := _CALLOUT_RE.match(line):
            rows.append({"unit_id": uid, "title": title, "admonition_type": match.group(1).casefold(), "marker_style": "blockquote_callout", "line_number": line_number, "title_text": field_value(match.group(2))})
        elif match := _COLON_RE.match(line):
            rows.append({"unit_id": uid, "title": title, "admonition_type": match.group(1).casefold(), "marker_style": "colon_fence", "line_number": line_number, "title_text": field_value(match.group(2))})
    return rows
