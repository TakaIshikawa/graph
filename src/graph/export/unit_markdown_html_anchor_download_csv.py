"""CSV export for Markdown-embedded HTML anchor download metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, domain, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "href", "download", "rel", "target", "type", "hreflang", "text_preview", "domain"]
_ANCHOR_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a\s*>|<a\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_DOWNLOAD_TYPES = ("download", "attachment", "octet-stream", "zip", "pdf", "msword", "spreadsheet", "presentation", "tar", "gzip")


def export_units_to_markdown_html_anchor_download_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["href"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _ANCHOR_RE.finditer(content):
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        mime_type = values.get("type", "")
        if "download" not in values and not any(token in mime_type.casefold() for token in _DOWNLOAD_TYPES):
            continue
        href = values.get("href", "")
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "href": href,
                "download": values.get("download", ""),
                "rel": values.get("rel", ""),
                "target": values.get("target", ""),
                "type": mime_type,
                "hreflang": values.get("hreflang", ""),
                "text_preview": preview(match.group("body") or ""),
                "domain": domain(href),
            }
        )
    return rows
