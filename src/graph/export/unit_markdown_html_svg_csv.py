"""CSV export for Markdown-embedded inline SVG blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "id", "class", "width", "height", "viewbox", "role", "aria_label", "title_text", "desc_text", "child_element_count", "has_external_reference"]
_SVG_RE = re.compile(r"<svg\b(?P<attrs>[^>]*)>(?P<body>.*?)</svg\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<([A-Za-z][\w:.-]*)\b[^>]*>", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title\b[^>]*>(?P<body>.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
_DESC_RE = re.compile(r"<desc\b[^>]*>(?P<body>.*?)</desc\s*>", re.IGNORECASE | re.DOTALL)
_EXT_REF_RE = re.compile(r"""(?:href|xlink:href)\s*=\s*["']https?://""", re.IGNORECASE)


def export_units_to_markdown_html_svg_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    for match in _SVG_RE.finditer(content):
        values = attrs(match.group("attrs"))
        body = match.group("body")
        title = _TITLE_RE.search(body)
        desc = _DESC_RE.search(body)
        rows.append({**context, "line_number": line_number(content, match.start()), "id": values.get("id", ""), "class": values.get("class", ""), "width": values.get("width", ""), "height": values.get("height", ""), "viewbox": values.get("viewbox", ""), "role": values.get("role", ""), "aria_label": values.get("aria-label", ""), "title_text": preview(title.group("body") if title else ""), "desc_text": preview(desc.group("body") if desc else ""), "child_element_count": len(_TAG_RE.findall(body)), "has_external_reference": str(bool(_EXT_REF_RE.search(body))).lower()})
    return rows
