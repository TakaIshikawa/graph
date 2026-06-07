"""CSV export for Markdown-embedded HTML media track elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, domain, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "parent_tag", "parent_src", "src", "domain", "kind", "srclang", "label", "default", "track_index"]
_MEDIA_RE = re.compile(r"<(?P<tag>audio|video)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)\s*>", re.IGNORECASE | re.DOTALL)
_TRACK_RE = re.compile(r"<track\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)
_SOURCE_RE = re.compile(r"<source\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)


def export_units_to_markdown_html_track_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["parent_tag"]), int(row["track_index"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for media in _MEDIA_RE.finditer(content):
        media_values = attrs(media.group("attrs"))
        body = media.group("body")
        first_source = _SOURCE_RE.search(body)
        parent_src = media_values.get("src", "") or (attrs(first_source.group("attrs")).get("src", "") if first_source else "")
        for index, track in enumerate(_TRACK_RE.finditer(body), start=1):
            values = attrs(track.group("attrs"))
            src = values.get("src", "")
            rows.append({**context, "line_number": line_number(content, media.start()) + body.count("\n", 0, track.start()), "parent_tag": media.group("tag").casefold(), "parent_src": parent_src, "src": src, "domain": domain(src), "kind": values.get("kind", ""), "srclang": values.get("srclang", ""), "label": values.get("label", ""), "default": bool_attr(values, "default"), "track_index": index})
    return rows
