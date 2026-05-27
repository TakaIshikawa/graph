"""CSV export for inline Markdown code spans."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "code", "delimiter_length", "line_number", "start_column", "context"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TICK_RE = re.compile(r"`+")


def export_units_to_markdown_inline_code_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["start_column"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for code, length, start_column in _inline_spans(line):
            rows.append({"unit_id": uid, "title": title, "code": field_value(code), "delimiter_length": length, "line_number": line_number, "start_column": start_column, "context": field_value(line)})
    return rows


def _inline_spans(line: str) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    pos = 0
    while match := _TICK_RE.search(line, pos):
        delimiter = match.group(0)
        end = line.find(delimiter, match.end())
        if end == -1:
            break
        code = line[match.end() : end]
        if code:
            rows.append((code, len(delimiter), match.start() + 1))
        pos = end + len(delimiter)
    return rows
