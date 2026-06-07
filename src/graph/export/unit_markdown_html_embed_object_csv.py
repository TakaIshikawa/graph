"""CSV export for Markdown-embedded HTML embed, object, and param elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, domain, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "src_or_data", "type", "width", "height", "name", "param_name", "param_value", "fallback_preview", "domain"]
_OBJECT_RE = re.compile(r"<object\b(?P<attrs>[^>]*)>(?P<body>.*?)</object\s*>", re.IGNORECASE | re.DOTALL)
_EMBED_RE = re.compile(r"<embed\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)
_PARAM_RE = re.compile(r"<param\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)


def export_units_to_markdown_html_embed_object_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["param_name"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    consumed: list[tuple[int, int]] = []
    for match in _OBJECT_RE.finditer(content):
        consumed.append(match.span())
        values = attrs(match.group("attrs"))
        body = match.group("body")
        src = values.get("data", "")
        base = {**context, "line_number": line_number(content, match.start()), "tag": "object", "src_or_data": src, "type": values.get("type", ""), "width": values.get("width", ""), "height": values.get("height", ""), "name": values.get("name", ""), "fallback_preview": preview(_PARAM_RE.sub(" ", body)), "domain": domain(src)}
        params = list(_PARAM_RE.finditer(body))
        if not params:
            rows.append({**base, "param_name": "", "param_value": ""})
        for param in params:
            param_values = attrs(param.group("attrs"))
            rows.append({**base, "param_name": param_values.get("name", ""), "param_value": param_values.get("value", "")})
    for match in _EMBED_RE.finditer(content):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        values = attrs(match.group("attrs"))
        src = values.get("src", "")
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": "embed", "src_or_data": src, "type": values.get("type", ""), "width": values.get("width", ""), "height": values.get("height", ""), "name": values.get("name", ""), "param_name": "", "param_value": "", "fallback_preview": "", "domain": domain(src)})
    return rows
