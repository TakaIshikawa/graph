"""CSV export for Markdown admonition and callout blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "syntax", "kind", "marker", "start_line", "first_text"]
_ADMONITION_RE = re.compile(r"^\s{0,3}([!?]{3})\s+([A-Za-z][\w-]*)(?:\s+(.+))?\s*$")
_CALLOUT_RE = re.compile(r"^\s{0,3}>\s*\[!([A-Za-z][\w-]*)\][+-]?\s*(.*)$")


def export_units_to_markdown_admonition_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["start_line"]), sort_key(row["kind"])))
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
    for line_number, line in enumerate(lines, start=1):
        if match := _ADMONITION_RE.match(line):
            first_text = field_value(match.group(3)) or _next_indented_text(lines, line_number)
            rows.append({"unit_id": uid, "title": title, "syntax": "admonition", "kind": match.group(2).casefold(), "marker": match.group(0).strip(), "start_line": line_number, "first_text": first_text})
        elif match := _CALLOUT_RE.match(line):
            first_text = field_value(match.group(2)) or _next_callout_text(lines, line_number)
            rows.append({"unit_id": uid, "title": title, "syntax": "obsidian_callout", "kind": match.group(1).casefold(), "marker": match.group(0).strip(), "start_line": line_number, "first_text": first_text})
    return rows


def _next_indented_text(lines: list[str], line_number: int) -> str:
    for line in lines[line_number:]:
        text = field_value(line)
        if text:
            return text
    return ""


def _next_callout_text(lines: list[str], line_number: int) -> str:
    for line in lines[line_number:]:
        stripped = line.strip()
        if not stripped.startswith(">"):
            break
        text = field_value(stripped[1:])
        if text:
            return text
    return ""
