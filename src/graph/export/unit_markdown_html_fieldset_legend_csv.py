"""CSV export for Markdown-embedded HTML fieldset and legend elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "fieldset_id", "fieldset_class", "disabled", "name", "legend_text", "control_count", "input_count", "select_count", "textarea_count"]
_FIELDSET_RE = re.compile(r"<fieldset\b(?P<attrs>[^>]*)>(?P<body>.*?)</fieldset\s*>", re.IGNORECASE | re.DOTALL)
_LEGEND_RE = re.compile(r"<legend\b[^>]*>(?P<body>.*?)</legend\s*>", re.IGNORECASE | re.DOTALL)
_CONTROL_RE = re.compile(r"<(?P<tag>input|select|textarea|button)\b", re.IGNORECASE)


def export_units_to_markdown_html_fieldset_legend_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["fieldset_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _FIELDSET_RE.finditer(content):
        body = match.group("body")
        values = attrs(match.group("attrs"))
        controls = [control.group("tag").casefold() for control in _CONTROL_RE.finditer(body)]
        legend = _LEGEND_RE.search(body)
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "fieldset_id": values.get("id", ""),
                "fieldset_class": values.get("class", ""),
                "disabled": bool_attr(values, "disabled"),
                "name": values.get("name", ""),
                "legend_text": preview(legend.group("body")) if legend else "",
                "control_count": len(controls),
                "input_count": controls.count("input"),
                "select_count": controls.count("select"),
                "textarea_count": controls.count("textarea"),
            }
        )
    return rows
