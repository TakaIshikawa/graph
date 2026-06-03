"""CSV export for GFM strikethrough spans in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "text", "line_number", "column_number", "character_count"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_STRIKETHROUGH_RE = re.compile(r"~~([^~\n]+?)~~")
_CODE_SPAN_RE = re.compile(r"(`+)(.*?)\1")


def export_units_to_markdown_strikethrough_span_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per double-tilde strikethrough span."""
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["column_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        code_spans = [(match.start(), match.end()) for match in _CODE_SPAN_RE.finditer(line)]
        for match in _STRIKETHROUGH_RE.finditer(line):
            if _overlaps_any(match.start(), match.end(), code_spans):
                continue
            text = field_value(match.group(1))
            rows.append({"unit_id": uid, "text": text, "line_number": line_number, "column_number": match.start() + 1, "character_count": len(text)})
    return rows


def _overlaps_any(match_start: int, match_end: int, spans: list[tuple[int, int]]) -> bool:
    return any(match_start < span_end and span_start < match_end for span_start, span_end in spans)
