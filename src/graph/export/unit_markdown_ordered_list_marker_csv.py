"""CSV export for Markdown ordered-list markers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "marker", "delimiter", "number", "indent"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ITEM_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<number>\d+)(?P<delimiter>[.)])\s+")


def export_unit_markdown_ordered_list_markers_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["indent"]), int(row["number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _ITEM_RE.match(line)
        if match:
            number = int(match.group("number"))
            delimiter = match.group("delimiter")
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "marker": f"{number}{delimiter}", "delimiter": delimiter, "number": number, "indent": len(match.group("indent").replace("\t", "    "))})
    return rows
