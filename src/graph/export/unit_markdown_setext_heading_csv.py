"""CSV export for Setext-style Markdown headings."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "level", "text", "underline"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_unit_markdown_setext_heading_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    previous: tuple[int, str] | None = None
    for line_number, line in _content_lines(str(get(unit, "content") or "")):
        level = _underline_level(line)
        if level and previous and previous[1].strip():
            rows.append({"unit_id": uid, "title": title, "line_number": previous[0], "level": level, "text": previous[1].strip(), "underline": line.strip()})
            previous = None
            continue
        previous = (line_number, line) if line.strip() else None
    return rows


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows


def _underline_level(line: str) -> int:
    text = line.strip()
    if len(text) < 2:
        return 0
    if set(text) == {"="}:
        return 1
    if set(text) == {"-"}:
        return 2
    return 0
