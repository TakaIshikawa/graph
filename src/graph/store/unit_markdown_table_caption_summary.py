"""Summarize Markdown table captions."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, unit_id

_CAPTION_RE = re.compile(r"^\s*(?:Table\s*:\s*|:\s+)(?P<caption>.+)$", re.IGNORECASE)


def summarize_unit_markdown_table_captions(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total = captioned = 0
    positions: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        lines = str(get(unit, "content") or "").splitlines()
        for start in _table_starts(lines):
            total += 1
            caption, position = _caption(lines, start)
            if caption:
                captioned += 1
                positions[position] += 1
            if len(samples) < sample_limit:
                samples.append({"unit_id": unit_id(unit), "table_start_line": start + 1, "caption": caption, "caption_position": position})
    return {"total_tables": total, "captioned_tables": captioned, "uncaptioned_tables": total - captioned, "caption_position_counts": [{"position": key, "count": positions[key]} for key in sorted(positions)], "samples": samples}


def _table_starts(lines: list[str]) -> list[int]:
    starts: list[int] = []
    index = 0
    while index < len(lines) - 1:
        if _pipe_cells(lines[index]) >= 2 and _is_separator(lines[index + 1]):
            starts.append(index)
            index += 2
            while index < len(lines) and _pipe_cells(lines[index]) >= 2:
                index += 1
            continue
        index += 1
    return starts


def _caption(lines: list[str], start: int) -> tuple[str, str]:
    if start > 0 and (match := _CAPTION_RE.match(lines[start - 1])):
        return field_value(match.group("caption")), "preceding"
    end = start + 2
    while end < len(lines) and _pipe_cells(lines[end]) >= 2:
        end += 1
    if end < len(lines) and (match := _CAPTION_RE.match(lines[end])):
        return field_value(match.group("caption")), "following"
    return "", "none"


def _pipe_cells(line: str) -> int:
    return len(line.strip().strip("|").split("|")) if "|" in line else 0


def _is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(cell and set(cell) <= {"-", ":"} and "-" in cell for cell in cells)
