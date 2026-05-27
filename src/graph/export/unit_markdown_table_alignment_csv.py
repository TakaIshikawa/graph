"""CSV export for Markdown pipe table column alignments."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "table_start_line", "column_index", "alignment", "delimiter_cell"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_markdown_table_alignment_csv(
    units: Iterable[Mapping[str, Any] | object], path_or_file: str | Path | TextIO | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["table_start_line"]), int(row["column_index"])))
    text = render_csv(rows, _FIELDNAMES)
    if path_or_file is None:
        return text
    if hasattr(path_or_file, "write"):
        path_or_file.write(text)
        return {"unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": len(text.encode("utf-8"))}
    output_path, bytes_written = write_csv(path_or_file, text)
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
        for column_index, cell in enumerate(delimiter_cells, start=1):
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "table_start_line": line_number,
                    "column_index": column_index,
                    "alignment": _alignment(cell),
                    "delimiter_cell": field_value(cell),
                }
            )
    return rows


def _content_lines(unit: Mapping[str, Any] | object) -> list[tuple[int, str]]:
    kept: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append((line_number, line))
    return kept


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _delimiter_cells(line: str) -> list[str]:
    cells = _cells(line)
    if len(cells) < 2:
        return []
    return cells if all(_is_delimiter_cell(cell) for cell in cells) else []


def _is_delimiter_cell(cell: str) -> bool:
    compact = cell.strip()
    return bool(compact) and "-" in compact and set(compact) <= {"-", ":"}


def _alignment(cell: str) -> str:
    compact = cell.strip()
    left = compact.startswith(":")
    right = compact.endswith(":")
    if left and right:
        return "center"
    if left:
        return "left"
    if right:
        return "right"
    return "default"
