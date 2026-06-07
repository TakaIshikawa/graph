"""CSV export for Markdown-embedded HTML noscript blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "content_preview", "contains_link", "contains_image", "contains_form", "nested_tag_count", "empty_content"]
_NOSCRIPT_RE = re.compile(r"<noscript\b[^>]*>(?P<body>.*?)</noscript\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<([A-Za-z][\w:.-]*)\b[^>]*>", re.IGNORECASE)


def export_units_to_markdown_html_noscript_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _NOSCRIPT_RE.finditer(content):
        body = match.group("body")
        tags = [tag.casefold() for tag in _TAG_RE.findall(body)]
        text = preview(body)
        rows.append({**context, "line_number": line_number(content, match.start()), "content_preview": text, "contains_link": str("a" in tags).lower(), "contains_image": str("img" in tags or "picture" in tags).lower(), "contains_form": str("form" in tags).lower(), "nested_tag_count": len(tags), "empty_content": str(not text).lower()})
    return rows
