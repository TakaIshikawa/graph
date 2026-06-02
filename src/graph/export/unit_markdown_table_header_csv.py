"""CSV export for Markdown pipe table header cells."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "column_index", "header_text", "is_empty"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_unit_markdown_table_headers_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["column_index"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    lines = _content_lines(unit)
    for index in range(len(lines) - 1):
        line_number, header = lines[index]
        _delimiter_number, delimiter = lines[index + 1]
        header_cells = _cells(header)
        delimiter_cells = _delimiter_cells(delimiter)
        if len(header_cells) < 2 or len(header_cells) != len(delimiter_cells):
            continue
        for column_index, cell in enumerate(header_cells, start=1):
            header_text = field_value(cell.replace(r"\|", "|"))
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "line_number": line_number,
                    "column_index": column_index,
                    "header_text": header_text,
                    "is_empty": "true" if not header_text else "false",
                }
            )
    return rows


def _content_lines(unit: Mapping[str, Any] | object) -> list[tuple[int, str]]:
    kept: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append((line_number, line))
    return kept


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in _split_unescaped_pipes(stripped)]


def _split_unescaped_pipes(line: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line:
        if char == "|" and not escaped:
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
        escaped = char == "\\" and not escaped
    cells.append("".join(current))
    return cells


def _delimiter_cells(line: str) -> list[str]:
    cells = _cells(line)
    return cells if len(cells) >= 2 and all(_is_delimiter_cell(cell) for cell in cells) else []


def _is_delimiter_cell(cell: str) -> bool:
    compact = cell.strip()
    return bool(re.fullmatch(r":?-{3,}:?", compact))
