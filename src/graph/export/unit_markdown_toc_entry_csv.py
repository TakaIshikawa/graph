"""CSV export for Markdown table-of-contents list entries."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "entry_text", "fragment", "list_marker", "indent", "line_number"]
_TOC_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>(?:[-+*])|(?:\d+[.)]))[ \t]+\[(?P<text>[^\]\n]+)]\(#(?P<fragment>[^)\s]+)\)")


def export_unit_markdown_toc_entry_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["indent"]), sort_key(row["fragment"])))
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
        match = _TOC_RE.match(line)
        if match:
            rows.append({
                "unit_id": uid,
                "title": title,
                "entry_text": field_value(match.group("text")),
                "fragment": field_value(match.group("fragment")),
                "list_marker": field_value(match.group("marker")),
                "indent": len(match.group("indent").replace("\t", "    ")),
                "line_number": line_number,
            })
    return rows
