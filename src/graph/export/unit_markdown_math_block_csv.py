"""CSV export for unit-level Markdown display math blocks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = [
    "unit_id",
    "title",
    "opening_line",
    "closing_line",
    "expression_line_count",
    "blank_line_count",
    "unterminated",
]


def export_unit_markdown_math_blocks_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["opening_line"]), int(row["closing_line"] or 0)))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    unit_title = field_value(get(unit, "title") or metadata(unit).get("title"))
    current: dict[str, int] | None = None

    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if line.strip() == "$$":
            if current is None:
                current = {"opening_line": line_number, "expression_line_count": 0, "blank_line_count": 0}
            else:
                rows.append(_row(unit, unit_title, current, line_number, False))
                current = None
            continue
        if current is not None:
            current["expression_line_count"] += 1
            if not line.strip():
                current["blank_line_count"] += 1

    if current is not None:
        rows.append(_row(unit, unit_title, current, "", True))
    return rows


def _row(
    unit: Mapping[str, Any] | object,
    title: str,
    block: dict[str, int],
    closing_line: int | str,
    unterminated: bool,
) -> dict[str, str | int]:
    return {
        "unit_id": unit_id(unit),
        "title": title,
        "opening_line": block["opening_line"],
        "closing_line": closing_line,
        "expression_line_count": block["expression_line_count"],
        "blank_line_count": block["blank_line_count"],
        "unterminated": "true" if unterminated else "false",
    }
