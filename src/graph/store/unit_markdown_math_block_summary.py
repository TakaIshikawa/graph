"""Summarize display math blocks delimited by standalone dollar markers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_markdown_math_blocks(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    """Summarize display math blocks delimited by standalone ``$$`` lines."""
    rows: list[dict[str, Any]] = []
    total_units = total_blocks = unterminated_total = 0
    for index, unit in enumerate(units):
        total_units += 1
        uid = unit_id(unit) or str(index)
        block_count, first_line, max_lines, unterminated = _counts(_content(unit))
        total_blocks += block_count
        unterminated_total += unterminated
        rows.append({
            "unit_id": uid,
            "math_block_count": block_count,
            "first_line_number": first_line,
            "max_block_line_count": max_lines,
            "unterminated_block_count": unterminated,
        })
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": total_units, "total_math_blocks": total_blocks, "unterminated_block_count": unterminated_total, "units": rows}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _counts(content: str) -> tuple[int, int, int, int]:
    in_code = in_math = False
    block_count = first_line = max_lines = unterminated = 0
    start_line = line_count = 0
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not in_math and _FENCE_RE.match(line):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.strip() != "$$":
            if in_math:
                line_count += 1
            continue
        if in_math:
            max_lines = max(max_lines, line_count)
            in_math = False
            line_count = 0
        else:
            block_count += 1
            first_line = first_line or line_number
            start_line = line_number
            in_math = True
            line_count = 0
    if in_math:
        unterminated = 1
        max_lines = max(max_lines, line_count)
        first_line = first_line or start_line
    return block_count, first_line, max_lines, unterminated
