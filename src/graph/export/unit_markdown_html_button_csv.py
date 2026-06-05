"""CSV export for Markdown-embedded HTML button elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "id", "name", "type", "value_present", "disabled", "formaction", "formmethod", "formtarget", "text_preview", "has_html_content"]
_BUTTON_RE = re.compile(r"<button\b(?P<attrs>[^>]*)>(?P<body>.*?)</button\s*>|<button\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def export_units_to_markdown_html_button_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["id"]), sort_key(row["name"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _BUTTON_RE.finditer(content):
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        body = match.group("body") or ""
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "id": values.get("id", ""),
                "name": values.get("name", ""),
                "type": values.get("type", ""),
                "value_present": bool_attr(values, "value"),
                "disabled": bool_attr(values, "disabled"),
                "formaction": values.get("formaction", ""),
                "formmethod": values.get("formmethod", ""),
                "formtarget": values.get("formtarget", ""),
                "text_preview": preview(body),
                "has_html_content": str(bool(_TAG_RE.search(body))).lower(),
            }
        )
    return rows
