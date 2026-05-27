"""Summarize shape details for markdown pipe tables."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id


def summarize_unit_markdown_table_shapes(units: Iterable[Any]) -> dict[str, Any]:
    total_units = total_tables = 0
    table_shapes: list[dict[str, Any]] = []
    row_buckets = Counter()
    column_buckets = Counter()
    malformed_units: set[str] = set()

    for index, unit in enumerate(units):
        total_units += 1
        identifier = unit_id(unit) or str(index)
        tables, malformed = _tables(_content(unit))
        if malformed:
            malformed_units.add(identifier)
        for table in tables:
            total_tables += 1
            table_shapes.append({"unit_id": identifier, **table})
            row_buckets[_row_bucket(table["rows"])] += 1
            column_buckets[table["columns"]] += 1

    return {
        "total_units": total_units,
        "total_tables": total_tables,
        "table_shapes": table_shapes,
        "row_bucket_counts": [{"bucket": key, "count": row_buckets[key]} for key in ("0", "1-2", "3-5", "6+")],
        "column_counts": [{"columns": key, "count": column_buckets[key]} for key in sorted(column_buckets)],
        "malformed_units": sorted(malformed_units, key=sort_key),
    }


def _tables(content: str) -> tuple[list[dict[str, int]], bool]:
    lines = content.splitlines()
    tables: list[dict[str, int]] = []
    malformed = False
    index = 0
    while index < len(lines) - 1:
        columns = _pipe_cells(lines[index])
        if columns >= 2:
            if _is_separator(lines[index + 1]):
                rows = 0
                cursor = index + 2
                while cursor < len(lines) and _pipe_cells(lines[cursor]) >= 2:
                    rows += 1
                    cursor += 1
                tables.append({"rows": rows, "columns": columns})
                index = cursor
                continue
            if _looks_separator_candidate(lines[index + 1]):
                malformed = True
        index += 1
    return tables, malformed


def _pipe_cells(line: str) -> int:
    stripped = line.strip()
    return len(stripped.strip("|").split("|")) if "|" in stripped else 0


def _is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(cell and set(cell) <= {"-", ":"} and "-" in cell for cell in cells)


def _looks_separator_candidate(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and "-" in stripped


def _row_bucket(rows: int) -> str:
    if rows == 0:
        return "0"
    if rows <= 2:
        return "1-2"
    if rows <= 5:
        return "3-5"
    return "6+"


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
