"""CSV export for Markdown-embedded HTML map area elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, domain, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_path",
    "source",
    "line_number",
    "map_name",
    "map_id",
    "shape",
    "coords",
    "href",
    "target",
    "alt",
    "download",
    "rel",
    "media",
    "type",
    "hreflang",
    "referrerpolicy",
    "ping_present",
    "nohref",
    "domain",
]
_MAP_RE = re.compile(r"<map\b(?P<attrs>[^>]*)>(?P<body>.*?)</map\s*>", re.IGNORECASE | re.DOTALL)
_AREA_RE = re.compile(r"<area\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)


def export_units_to_markdown_html_map_area_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["map_name"]), sort_key(row["href"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for map_match in _MAP_RE.finditer(content):
        map_values = attrs(map_match.group("attrs"))
        body = map_match.group("body")
        for area_match in _AREA_RE.finditer(body):
            area_values = attrs(area_match.group("attrs"))
            href = area_values.get("href", "")
            rows.append(
                {
                    **context,
                    "line_number": line_number(content, map_match.start()) + body.count("\n", 0, area_match.start()),
                    "map_name": map_values.get("name", ""),
                    "map_id": map_values.get("id", ""),
                    "shape": area_values.get("shape", ""),
                    "coords": area_values.get("coords", ""),
                    "href": href,
                    "target": area_values.get("target", ""),
                    "alt": area_values.get("alt", ""),
                    "download": area_values.get("download", ""),
                    "rel": area_values.get("rel", ""),
                    "media": area_values.get("media", ""),
                    "type": area_values.get("type", ""),
                    "hreflang": area_values.get("hreflang", ""),
                    "referrerpolicy": area_values.get("referrerpolicy", ""),
                    "ping_present": bool_attr(area_values, "ping"),
                    "nohref": bool_attr(area_values, "nohref"),
                    "domain": domain(href),
                }
            )
    return rows
