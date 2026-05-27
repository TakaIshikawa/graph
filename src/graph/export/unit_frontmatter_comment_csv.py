"""CSV export for comments in leading YAML frontmatter."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "comment", "inline", "field_key"]
_FIELD_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*)$")


def export_unit_frontmatter_comment_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["field_key"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in _frontmatter_lines(str(get(unit, "content") or "")):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "comment": stripped[1:].strip(), "inline": "false", "field_key": ""})
            continue
        match = _FIELD_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        comment = _inline_comment(value)
        if comment is not None:
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "comment": comment.strip(), "inline": "true", "field_key": key})
    return rows


def _frontmatter_lines(content: str) -> list[tuple[int, str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    rows: list[tuple[int, str]] = []
    for offset, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            return rows
        rows.append((offset, line))
    return []


def _inline_comment(value: str) -> str | None:
    quote = ""
    for index, char in enumerate(value):
        if char in {"'", '"'} and (index == 0 or value[index - 1] != "\\"):
            quote = "" if quote == char else char if not quote else quote
        if char == "#" and not quote and (index == 0 or value[index - 1].isspace()):
            return value[index + 1 :]
    return None
