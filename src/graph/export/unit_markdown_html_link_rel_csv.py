"""CSV export for Markdown-embedded HTML link rel metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, domain, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "href", "rel", "as", "type", "media", "sizes", "hreflang", "integrity", "crossorigin", "referrerpolicy", "domain"]
_LINK_RE = re.compile(r"<link\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_link_rel_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["rel"]), sort_key(row["href"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _LINK_RE.finditer(content):
        values = attrs(match.group("attrs"))
        href = values.get("href", "")
        rows.append({**context, "line_number": line_number(content, match.start()), "href": href, "rel": values.get("rel", ""), "as": values.get("as", ""), "type": values.get("type", ""), "media": values.get("media", ""), "sizes": values.get("sizes", ""), "hreflang": values.get("hreflang", ""), "integrity": values.get("integrity", ""), "crossorigin": values.get("crossorigin", ""), "referrerpolicy": values.get("referrerpolicy", ""), "domain": domain(href)})
    return rows
