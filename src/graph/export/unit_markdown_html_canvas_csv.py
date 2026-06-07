"""CSV export for Markdown-embedded HTML canvas elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "id", "class", "width", "height", "fallback_preview", "has_fallback_content", "aria_label", "role", "nested_html_present"]
_CANVAS_RE = re.compile(r"<canvas\b(?P<attrs>[^>]*)>(?P<body>.*?)</canvas\s*>|<canvas\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[A-Za-z][^>]*>")


def export_units_to_markdown_html_canvas_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _CANVAS_RE.finditer(content):
        body = match.group("body") or ""
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        fallback = preview(body)
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
                "width": values.get("width", ""),
                "height": values.get("height", ""),
                "fallback_preview": fallback,
                "has_fallback_content": str(bool(fallback)).lower(),
                "aria_label": values.get("aria-label", ""),
                "role": values.get("role", ""),
                "nested_html_present": str(bool(_TAG_RE.search(body))).lower(),
            }
        )
    return rows
