"""CSV export for Markdown-embedded HTML output elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "name", "for", "form", "text_preview", "has_value_text", "aria_live", "role", "class", "id", "nested_html_present"]
_OUTPUT_RE = re.compile(r"<output\b(?P<attrs>[^>]*)>(?P<body>.*?)</output\s*>|<output\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[A-Za-z][^>]*>")


def export_units_to_markdown_html_output_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["name"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _OUTPUT_RE.finditer(content):
        body = match.group("body") or ""
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        text = preview(body)
        rows.append({**context, "line_number": line_number(content, match.start()), "name": values.get("name", ""), "for": values.get("for", ""), "form": values.get("form", ""), "text_preview": text, "has_value_text": str(bool(text)).lower(), "aria_live": values.get("aria-live", ""), "role": values.get("role", ""), "class": values.get("class", ""), "id": values.get("id", ""), "nested_html_present": str(bool(_TAG_RE.search(body))).lower()})
    return rows
