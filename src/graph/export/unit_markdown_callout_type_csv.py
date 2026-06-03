"""CSV export for Obsidian callout markers in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "callout_type", "fold_marker", "callout_title", "blockquote_depth"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_CALLOUT_RE = re.compile(r"^\s*(?P<markers>(?:>\s*)+)\[!(?P<type>[A-Za-z][A-Za-z0-9_-]*)\](?P<fold>[+-])?\s*(?P<title>.*)$")


def export_units_to_markdown_callout_type_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["callout_type"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    meta = metadata(unit)
    title = field_value(get(unit, "title") or meta.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_id") or meta.get("source") or meta.get("source_id"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or meta.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _CALLOUT_RE.match(line)
        if match:
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "source": source,
                    "line_number": line_number,
                    "callout_type": match.group("type").casefold(),
                    "fold_marker": match.group("fold") or "",
                    "callout_title": field_value(match.group("title")),
                    "blockquote_depth": match.group("markers").count(">"),
                }
            )
    return rows
