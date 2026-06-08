"""CSV export for Markdown-embedded HTML details and summary elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "open", "summary_text_preview", "details_text_preview", "summary_count", "missing_summary", "id", "class"]
_DETAILS_RE = re.compile(r"<details\b(?P<attrs>[^>]*)>(?P<body>.*?)</details\s*>", re.IGNORECASE | re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary\b(?P<attrs>[^>]*)>(?P<body>.*?)</summary\s*>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_details_summary_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _DETAILS_RE.finditer(content):
        values = attrs(match.group("attrs"))
        summaries = list(_SUMMARY_RE.finditer(match.group("body") or ""))
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": "details", "open": bool_attr(values, "open"), "summary_text_preview": preview(summaries[0].group("body")) if summaries else "", "details_text_preview": preview(match.group("body") or ""), "summary_count": len(summaries), "missing_summary": str(not summaries).lower(), "id": values.get("id", ""), "class": values.get("class", "")})
        for summary in summaries:
            summary_values = attrs(summary.group("attrs"))
            rows.append({**context, "line_number": line_number(content, match.start() + summary.start()), "tag": "summary", "open": "", "summary_text_preview": preview(summary.group("body") or ""), "details_text_preview": "", "summary_count": "", "missing_summary": "", "id": summary_values.get("id", ""), "class": summary_values.get("class", "")})
    return rows
