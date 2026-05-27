"""CSV export for LaTeX-style math notation in unit content."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "inline_math_count", "block_math_count", "unterminated_block_count", "first_math_line", "longest_math_chars"]


def export_units_to_math_notation_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    inventory = _inventory(content)
    return {
        "unit_id": unit_id(unit),
        "inline_math_count": inventory["inline_math_count"],
        "block_math_count": inventory["block_math_count"],
        "unterminated_block_count": inventory["unterminated_block_count"],
        "first_math_line": inventory["first_math_line"],
        "longest_math_chars": inventory["longest_math_chars"],
    }


def _inventory(content: str) -> dict[str, int | str]:
    inline_count = block_count = unterminated = longest = 0
    first_line = ""
    in_block = False
    block_chars = 0

    for line_number, line in enumerate(content.splitlines(), start=1):
        marker_count = _block_marker_count(line)
        if marker_count:
            if not first_line:
                first_line = str(line_number)
            if in_block:
                block_count += 1
                longest = max(longest, block_chars)
                in_block = False
            else:
                in_block = True
                block_chars = 0
            if marker_count > 1:
                block_count += marker_count // 2
                in_block = bool(marker_count % 2)
            continue
        if in_block:
            block_chars += len(line)
            continue
        count, segment_longest = _inline_math(line)
        if count and not first_line:
            first_line = str(line_number)
        inline_count += count
        longest = max(longest, segment_longest)

    if in_block:
        unterminated = 1
        longest = max(longest, block_chars)

    return {
        "inline_math_count": inline_count,
        "block_math_count": block_count,
        "unterminated_block_count": unterminated,
        "first_math_line": first_line,
        "longest_math_chars": longest,
    }


def _block_marker_count(line: str) -> int:
    count = 0
    index = 0
    while index < len(line):
        if line.startswith("$$", index) and not _escaped(line, index):
            count += 1
            index += 2
        else:
            index += 1
    return count


def _inline_math(text: str) -> tuple[int, int]:
    count = longest = 0
    start: int | None = None
    index = 0
    while index < len(text):
        if text.startswith("$$", index) and not _escaped(text, index):
            index += 2
            continue
        if text[index] == "$" and not _escaped(text, index):
            if start is None:
                start = index
            else:
                value = text[start + 1 : index].strip()
                if value:
                    count += 1
                    longest = max(longest, len(value))
                start = None
        index += 1
    return count, longest


def _escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1
