"""CSV export for Markdown blockquote nesting depths."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "depth", "quoted_text", "is_blank_quote"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_markdown_blockquote_depth_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per Markdown blockquote line outside fenced code."""
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["depth"]), sort_key(row["quoted_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    meta = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or meta.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_project") or meta.get("source") or meta.get("source_project"))
    rows: list[dict[str, str | int]] = []
    in_fence = False

    for line_number, line in enumerate(_content(unit).splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        parsed = _blockquote_line(line)
        if parsed is None:
            continue
        depth, quoted_text = parsed
        rows.append(
            {
                "unit_id": uid,
                "title": title,
                "source": source,
                "line_number": line_number,
                "depth": depth,
                "quoted_text": field_value(quoted_text),
                "is_blank_quote": "true" if not quoted_text.strip() else "false",
            }
        )
    return rows


def _content(unit: Mapping[str, Any] | object) -> str:
    if isinstance(unit, str):
        return unit
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _blockquote_line(line: str) -> tuple[int, str] | None:
    stripped = line.lstrip()
    depth = 0
    while stripped.startswith(">"):
        depth += 1
        stripped = stripped[1:].lstrip()
    if not depth:
        return None
    return depth, stripped
