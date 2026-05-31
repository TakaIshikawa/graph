"""CSV export for titled Markdown link definitions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "label", "url", "link_title", "title_style"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^\s{0,3}\[([^\]\n]+)\]:\s*(\S+)(?:\s+(.*?))?\s*$")


def export_units_to_markdown_link_definition_title_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in _content_lines(str(get(unit, "content") or "")):
        match = _DEF_RE.match(line)
        if not match:
            continue
        parsed = _title(match.group(3) or "")
        if parsed is None:
            continue
        link_title, title_style = parsed
        rows.append({"unit_id": uid, "title": title, "line_number": line_number, "label": field_value(match.group(1)), "url": field_value(match.group(2).strip("<>")), "link_title": link_title, "title_style": title_style})
    return rows


def _title(text: str) -> tuple[str, str] | None:
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return field_value(text[1:-1]), "quoted"
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return field_value(text[1:-1]), "single_quoted"
    if len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        return field_value(text[1:-1]), "parenthesized"
    return None


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
