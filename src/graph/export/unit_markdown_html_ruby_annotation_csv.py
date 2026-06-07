"""CSV export for Markdown-embedded HTML ruby annotation usage."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "ruby_text", "rt_text", "rp_text", "annotation_count", "has_fallback_parentheses", "nested_html_present"]
_RUBY_RE = re.compile(r"<ruby\b[^>]*>(?P<body>.*?)</ruby\s*>", re.IGNORECASE | re.DOTALL)
_RT_RE = re.compile(r"<rt\b[^>]*>(?P<body>.*?)</rt\s*>", re.IGNORECASE | re.DOTALL)
_RP_RE = re.compile(r"<rp\b[^>]*>(?P<body>.*?)</rp\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<([A-Za-z][\w:.-]*)\b[^>]*>", re.IGNORECASE)


def export_units_to_markdown_html_ruby_annotation_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["ruby_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _RUBY_RE.finditer(content):
        body = match.group("body")
        rt_values = [preview(rt.group("body")) for rt in _RT_RE.finditer(body)]
        rp_values = [preview(rp.group("body")) for rp in _RP_RE.finditer(body)]
        ruby_body = _RP_RE.sub(" ", _RT_RE.sub(" ", body))
        tags = [tag.casefold() for tag in _TAG_RE.findall(body)]
        rows.append({**context, "line_number": line_number(content, match.start()), "ruby_text": preview(ruby_body), "rt_text": " | ".join(rt_values), "rp_text": " | ".join(rp_values), "annotation_count": len(rt_values), "has_fallback_parentheses": str("(" in rp_values and ")" in rp_values).lower(), "nested_html_present": str(any(tag not in {"rt", "rp"} for tag in tags)).lower()})
    return rows
