"""CSV export for Markdown-embedded HTML table structure elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "caption_text", "colgroup_count", "col_count", "thead_count", "tbody_count", "tfoot_count", "row_count"]
_TABLE_RE = re.compile(r"<table\b[^>]*>(?P<body>.*?)</table\s*>", re.IGNORECASE | re.DOTALL)
_CAPTION_RE = re.compile(r"<caption\b[^>]*>(?P<body>.*?)</caption\s*>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_table_structure_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["caption_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _TABLE_RE.finditer(content):
        body = match.group("body")
        caption = _CAPTION_RE.search(body)
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "caption_text": preview(caption.group("body")) if caption else "",
                "colgroup_count": _count("colgroup", body),
                "col_count": _count("col", body),
                "thead_count": _count("thead", body),
                "tbody_count": _count("tbody", body),
                "tfoot_count": _count("tfoot", body),
                "row_count": _count("tr", body),
            }
        )
    return rows


def _count(tag: str, body: str) -> int:
    return len(re.findall(rf"<{tag}\b", body, re.IGNORECASE))
