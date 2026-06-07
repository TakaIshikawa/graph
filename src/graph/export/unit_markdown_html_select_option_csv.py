"""CSV export for Markdown-embedded HTML select and option elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "select_name", "select_id", "multiple", "required", "disabled", "option_count", "selected_count", "empty_value_count", "option_values", "selected_labels_preview"]
_SELECT_RE = re.compile(r"<select\b(?P<attrs>[^>]*)>(?P<body>.*?)</select\s*>", re.IGNORECASE | re.DOTALL)
_OPTION_RE = re.compile(r"<option\b(?P<attrs>[^>]*)>(?P<body>.*?)</option\s*>|<option\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_select_option_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["select_name"]), sort_key(row["select_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _SELECT_RE.finditer(content):
        values = attrs(match.group("attrs"))
        options = [_option(option) for option in _OPTION_RE.finditer(match.group("body"))]
        selected = [option for option in options if option["selected"]]
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "select_name": values.get("name", ""),
                "select_id": values.get("id", ""),
                "multiple": bool_attr(values, "multiple"),
                "required": bool_attr(values, "required"),
                "disabled": bool_attr(values, "disabled"),
                "option_count": len(options),
                "selected_count": len(selected),
                "empty_value_count": sum(option["value"] == "" for option in options),
                "option_values": "|".join(option["value"] for option in options),
                "selected_labels_preview": "; ".join(option["label"] for option in selected)[:120],
            }
        )
    return rows


def _option(match: re.Match[str]) -> dict[str, str | bool]:
    values = attrs(match.group("attrs") or match.group("single_attrs") or "")
    label = preview(match.group("body") or "")
    return {"value": values.get("value", label), "label": values.get("label", label), "selected": "selected" in values}
