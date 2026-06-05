"""CSV export for Markdown-embedded HTML table cell scope metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "scope", "headers", "colspan", "rowspan", "abbr", "text_preview", "has_scope_or_headers"]
_CELL_RE = re.compile(r"<(?P<tag>th|td)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?:th|td)\s*>|<(?P<single_tag>th|td)\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_table_cell_scope_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["text_preview"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _CELL_RE.finditer(content):
        tag = (match.group("tag") or match.group("single_tag")).casefold()
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        has_linkage = "scope" in values or "headers" in values
        if tag != "th" and not has_linkage:
            continue
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": tag, "scope": values.get("scope", ""), "headers": values.get("headers", ""), "colspan": values.get("colspan", ""), "rowspan": values.get("rowspan", ""), "abbr": values.get("abbr", ""), "text_preview": preview(match.group("body") or ""), "has_scope_or_headers": str(has_linkage).lower()})
    return rows
