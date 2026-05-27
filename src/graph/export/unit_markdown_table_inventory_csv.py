"""CSV export for pipe-style Markdown table inventory by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "table_index", "start_line", "header_column_count", "data_row_count", "header_preview"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_markdown_table_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["table_index"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    lines = _content_lines(unit)
    index = 0
    while index < len(lines) - 1:
        line_number, header = lines[index]
        if _pipe_cells(header) >= 2 and _is_separator(lines[index + 1][1]):
            cursor = index + 2
            data_rows = 0
            while cursor < len(lines) and _pipe_cells(lines[cursor][1]) >= 2:
                data_rows += 1
                cursor += 1
            rows.append({
                "unit_id": unit_id(unit),
                "title": field_value(get(unit, "title")),
                "table_index": len(rows) + 1,
                "start_line": line_number,
                "header_column_count": _pipe_cells(header),
                "data_row_count": data_rows,
                "header_preview": " | ".join(_cells(header)),
            })
            index = cursor
            continue
        index += 1
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


def _pipe_cells(line: str) -> int:
    return len(_cells(line)) if "|" in line else 0


def _cells(line: str) -> list[str]:
    return [field_value(cell) for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(cell and set(cell) <= {"-", ":"} and "-" in cell for cell in cells)
