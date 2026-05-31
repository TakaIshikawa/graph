"""CSV export for Markdown footnote definitions in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "label", "definition", "line", "continued_lines"]
_DEF_RE = re.compile(r"^\[\^([^\]\n]+)\]:\s*(.*)$")


def export_unit_markdown_footnote_definition_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    lines = str(get(unit, "content") or "").splitlines()
    rows: list[dict[str, str | int]] = []
    index = 0
    while index < len(lines):
        match = _DEF_RE.match(lines[index])
        if not match:
            index += 1
            continue
        parts = [field_value(match.group(2))]
        start = index + 1
        continued = 0
        index += 1
        while index < len(lines) and (lines[index].startswith("    ") or lines[index].startswith("\t")):
            parts.append(field_value(lines[index]))
            continued += 1
            index += 1
        rows.append({"unit_id": uid, "title": title, "label": field_value(match.group(1)), "definition": " ".join(part for part in parts if part), "line": start, "continued_lines": continued})
    return rows
