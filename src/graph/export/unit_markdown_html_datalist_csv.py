"""CSV export for Markdown-embedded HTML datalist suggestions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "datalist_id", "option_value", "option_label", "option_text", "option_count", "id", "class", "row_type"]
_DATALIST_RE = re.compile(r"<datalist\b(?P<attrs>[^>]*)>(?P<body>.*?)</datalist\s*>|<datalist\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_OPTION_RE = re.compile(r"<option\b(?P<attrs>[^>]*)>(?P<body>.*?)</option\s*>|<option\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_datalist_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["datalist_id"]), sort_key(row["row_type"]), sort_key(row["option_value"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _DATALIST_RE.finditer(content):
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        body = match.group("body") or ""
        options = list(_OPTION_RE.finditer(body))
        base = {**context, "line_number": line_number(content, match.start()), "datalist_id": values.get("id", ""), "id": values.get("id", ""), "class": values.get("class", "")}
        rows.append({**base, "option_value": "", "option_label": "", "option_text": "", "option_count": len(options), "row_type": "datalist"})
        for option in options:
            option_values = attrs(option.group("attrs") or option.group("single_attrs") or "")
            rows.append({**base, "line_number": line_number(content, match.start() + option.start()), "option_value": option_values.get("value", ""), "option_label": option_values.get("label", ""), "option_text": preview(option.group("body") or ""), "option_count": "", "row_type": "option"})
    return rows
