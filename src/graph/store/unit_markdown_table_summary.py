"""Summarize GitHub-style markdown tables in unit content."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id


def summarize_unit_markdown_tables(units: Iterable[Any]) -> dict[str, Any]:
    total_units = units_with_tables = total_tables = 0
    malformed_units: set[str] = set()
    row_distribution: Counter[int] = Counter()

    for index, unit in enumerate(units):
        total_units += 1
        tables, malformed = _tables(_content(unit))
        identifier = unit_id(unit) or str(index)
        if tables:
            units_with_tables += 1
        if malformed:
            malformed_units.add(identifier)
        total_tables += len(tables)
        row_distribution.update(tables)

    return {
        "total_units": total_units,
        "total_tables": total_tables,
        "units_with_tables": units_with_tables,
        "row_count_distribution": [
            {"rows": rows, "count": row_distribution[rows]} for rows in sorted(row_distribution)
        ],
        "malformed_units": sorted(malformed_units, key=sort_key),
    }


def _tables(content: str) -> tuple[list[int], bool]:
    lines = content.splitlines()
    rows: list[int] = []
    malformed = False
    index = 0
    while index < len(lines) - 1:
        if _pipe_cells(lines[index]) >= 2:
            if _is_separator(lines[index + 1]):
                data_rows = 0
                cursor = index + 2
                while cursor < len(lines) and _pipe_cells(lines[cursor]) >= 2:
                    data_rows += 1
                    cursor += 1
                rows.append(data_rows)
                index = cursor
                continue
            if _looks_separator_candidate(lines[index + 1]):
                malformed = True
        index += 1
    return rows, malformed


def _pipe_cells(line: str) -> int:
    stripped = line.strip()
    if "|" not in stripped:
        return 0
    return len([cell for cell in stripped.strip("|").split("|")])


def _is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(cell and set(cell) <= {"-", ":"} and "-" in cell for cell in cells)


def _looks_separator_candidate(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and any(char == "-" for char in stripped)


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
