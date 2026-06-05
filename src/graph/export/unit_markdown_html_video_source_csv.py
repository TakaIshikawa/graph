"""CSV export for Markdown-embedded HTML video and source elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, domain, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "video_src", "source_src", "type", "poster", "controls", "autoplay", "muted", "loop", "preload", "width", "height", "domain"]
_VIDEO_RE = re.compile(r"<video\b(?P<attrs>[^>]*)>(?P<body>.*?)</video\s*>|<video\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_SOURCE_RE = re.compile(r"<source\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)


def export_units_to_markdown_html_video_source_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["video_src"]), sort_key(row["source_src"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _VIDEO_RE.finditer(content):
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        body = match.group("body") or ""
        video_src = values.get("src", "")
        source_matches = list(_SOURCE_RE.finditer(body))
        if video_src and not source_matches:
            rows.append(_row(context, line_number(content, match.start()), values, video_src, "", values.get("type", "")))
        for source_match in source_matches:
            source_values = attrs(source_match.group("attrs"))
            source_src = source_values.get("src", "")
            rows.append(_row(context, line_number(content, match.start()) + body.count("\n", 0, source_match.start()), values, video_src, source_src, source_values.get("type", "")))
    return rows


def _row(context: dict[str, str], line: int, video_attrs: Mapping[str, str], video_src: str, source_src: str, mime_type: str) -> dict[str, str | int]:
    url = source_src or video_src
    return {**context, "line_number": line, "video_src": video_src, "source_src": source_src, "type": mime_type, "poster": video_attrs.get("poster", ""), "controls": bool_attr(video_attrs, "controls"), "autoplay": bool_attr(video_attrs, "autoplay"), "muted": bool_attr(video_attrs, "muted"), "loop": bool_attr(video_attrs, "loop"), "preload": video_attrs.get("preload", ""), "width": video_attrs.get("width", ""), "height": video_attrs.get("height", ""), "domain": domain(url)}
