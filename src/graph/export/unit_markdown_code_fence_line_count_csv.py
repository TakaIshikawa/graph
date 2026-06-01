"""CSV inventory for Markdown fenced code block line counts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "opening_line", "closing_line", "info_string", "content_line_count", "blank_line_count", "unterminated"]
_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


def export_unit_markdown_code_fence_line_counts_to_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per Markdown fenced code block."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _fence_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["opening_line"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _fence_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    current: dict[str, Any] | None = None
    lines = content.splitlines()
    for line_number, line in enumerate(lines, start=1):
        match = _FENCE_RE.match(line)
        if current is None:
            if match:
                marker = match.group(2)
                current = {"opening_line": line_number, "marker_char": marker[0], "marker_len": len(marker), "info_string": field_value(match.group(3)), "content": []}
            continue
        if match and match.group(2)[0] == current["marker_char"] and len(match.group(2)) >= current["marker_len"]:
            rows.append(_row(current, line_number, False))
            current = None
            continue
        current["content"].append(line)
    if current is not None:
        rows.append(_row(current, "", True))
    return rows


def _row(current: dict[str, Any], closing_line: int | str, unterminated: bool) -> dict[str, str | int]:
    content = current["content"]
    return {
        "opening_line": current["opening_line"],
        "closing_line": closing_line,
        "info_string": current["info_string"],
        "content_line_count": len(content),
        "blank_line_count": sum(1 for line in content if not line.strip()),
        "unterminated": "true" if unterminated else "false",
    }
