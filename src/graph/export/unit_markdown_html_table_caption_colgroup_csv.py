"""CSV export for Markdown-embedded HTML table caption and column elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "text_preview", "span", "column_count", "id", "class", "empty_caption"]
_TAG_RE = re.compile(r"<(?P<tag>caption|colgroup|col)\b(?P<attrs>[^>]*)(?:>(?P<body>.*?)</(?P=tag)\s*>|/?>)", re.IGNORECASE | re.DOTALL)
_COL_RE = re.compile(r"<col\b", re.IGNORECASE)
_COL_TAG_RE = re.compile(r"<col\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_table_caption_colgroup_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    for match in _TAG_RE.finditer(content):
        tag = match.group("tag").casefold()
        values = attrs(match.group("attrs") or "")
        text = preview(match.group("body") or "")
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": tag, "text_preview": text if tag == "caption" else "", "span": values.get("span", ""), "column_count": _column_count(tag, values, match.group("body") or ""), "id": values.get("id", ""), "class": values.get("class", ""), "empty_caption": str(tag == "caption" and not text).lower()})
        if tag == "colgroup":
            for col in _COL_TAG_RE.finditer(match.group("body") or ""):
                col_values = attrs(col.group("attrs") or "")
                rows.append({**context, "line_number": line_number(content, match.start() + col.start()), "tag": "col", "text_preview": "", "span": col_values.get("span", ""), "column_count": _span(col_values), "id": col_values.get("id", ""), "class": col_values.get("class", ""), "empty_caption": "false"})
    return rows


def _column_count(tag: str, values: Mapping[str, str], body: str) -> int | str:
    if tag == "col":
        return _span(values)
    if tag == "colgroup":
        cols = len(_COL_RE.findall(body))
        return cols if cols else _span(values)
    return ""


def _span(values: Mapping[str, str]) -> int:
    try:
        return max(1, int(values.get("span", "1")))
    except ValueError:
        return 1
