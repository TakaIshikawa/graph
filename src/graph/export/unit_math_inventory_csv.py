"""CSV export for Markdown math spans and blocks in unit content."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "display_math_block_count", "inline_math_span_count", "unterminated_display_math_count", "max_display_math_line_count"]


def export_units_to_math_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, int | str]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    body, blocks, unterminated, max_lines = _display_blocks(content)
    return {
        "unit_id": unit_id(unit),
        "display_math_block_count": blocks,
        "inline_math_span_count": _inline_count(body),
        "unterminated_display_math_count": unterminated,
        "max_display_math_line_count": max_lines,
    }


def _display_blocks(content: str) -> tuple[str, int, int, int]:
    kept: list[str] = []
    in_block = False
    block_count = 0
    line_count = 0
    max_lines = 0
    for line in content.splitlines():
        if line.strip() == "$$":
            if in_block:
                block_count += 1
                max_lines = max(max_lines, line_count)
                in_block = False
            else:
                in_block = True
                line_count = 0
            continue
        if in_block:
            line_count += 1
        else:
            kept.append(line)
    return "\n".join(kept), block_count, 1 if in_block else 0, max(max_lines, line_count if in_block else 0)


def _inline_count(text: str) -> int:
    count = 0
    index = 0
    while index < len(text):
        if text[index] != "$" or text[index : index + 2] == "$$" or (index + 1 < len(text) and text[index + 1].isdigit()):
            index += 1
            continue
        close = text.find("$", index + 1)
        if close == -1:
            index += 1
            continue
        if close > index + 1 and not text[close - 1].isspace():
            count += 1
        index = close + 1
    return count
