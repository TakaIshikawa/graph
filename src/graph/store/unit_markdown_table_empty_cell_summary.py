"""Summarize empty cells in markdown pipe tables."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id


def summarize_unit_markdown_table_empty_cells(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = units_with_empty_cells = total_empty_cells = 0
    column_counts: Counter[int] = Counter()
    samples: list[dict[str, Any]] = []

    for index, unit in enumerate(units):
        total_units += 1
        identifier = unit_id(unit) or str(index)
        unit_empty_cells = 0
        for line_number, row_text, column_position in _empty_cells(_content(unit)):
            unit_empty_cells += 1
            total_empty_cells += 1
            column_counts[column_position] += 1
            if len(samples) < limit:
                samples.append(
                    {
                        "unit_id": identifier,
                        "line_number": line_number,
                        "column_position": column_position,
                        "row_text": row_text,
                    }
                )
        if unit_empty_cells:
            units_with_empty_cells += 1

    samples.sort(key=lambda row: (sort_key(row["unit_id"]), row["line_number"], row["column_position"], row["row_text"]))
    most_common_column_position = None
    if column_counts:
        most_common_column_position = sorted(column_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {
        "total_units": total_units,
        "units_with_empty_cells": units_with_empty_cells,
        "total_empty_cells": total_empty_cells,
        "most_common_column_position": most_common_column_position,
        "samples": samples[:limit],
    }


def _empty_cells(content: str) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or not _is_table_row(line):
            continue
        cells = line.strip().strip("|").split("|")
        if _is_separator_cells(cells):
            continue
        for position, cell in enumerate(cells, start=1):
            if not cell.strip():
                rows.append((line_number, line.strip(), position))
    return rows


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and len(stripped.strip("|").split("|")) >= 2


def _is_separator_cells(cells: list[str]) -> bool:
    return all((cell := value.strip()) and set(cell) <= {"-", ":"} and "-" in cell for value in cells)


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
