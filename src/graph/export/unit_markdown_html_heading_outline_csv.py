"""CSV export for Markdown-embedded HTML heading outline metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "level", "text_preview", "id", "class", "empty_heading", "outline_jump_from_previous"]
_HEADING_RE = re.compile(r"<h(?P<level>[1-6])\b(?P<attrs>[^>]*)>(?P<body>.*?)</h(?P=level)\s*>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_heading_outline_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["level"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    previous_level: int | None = None
    for match in _HEADING_RE.finditer(content):
        values = attrs(match.group("attrs"))
        level = int(match.group("level"))
        text = preview(match.group("body") or "")
        rows.append({**context, "line_number": line_number(content, match.start()), "level": level, "text_preview": text, "id": values.get("id", ""), "class": values.get("class", ""), "empty_heading": str(not text).lower(), "outline_jump_from_previous": str(previous_level is not None and level > previous_level + 1).lower()})
        previous_level = level
    return rows
