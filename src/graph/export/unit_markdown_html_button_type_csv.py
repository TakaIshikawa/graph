"""CSV export for Markdown-embedded HTML button types."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "type", "normalized_type", "is_submit", "is_button", "is_reset", "disabled", "name", "value", "form", "id", "class", "text_preview"]
_BUTTON_RE = re.compile(r"<button\b(?P<attrs>[^>]*)>(?P<body>.*?)</button\s*>|<button\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_button_type_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["normalized_type"]), sort_key(row["id"])))
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
        raw_type = values.get("type", "")
        normalized = raw_type.casefold() if raw_type.casefold() in {"submit", "button", "reset"} else "submit"
        rows.append({**context, "line_number": line_number(content, match.start()), "type": raw_type, "normalized_type": normalized, "is_submit": str(normalized == "submit").lower(), "is_button": str(normalized == "button").lower(), "is_reset": str(normalized == "reset").lower(), "disabled": bool_attr(values, "disabled"), "name": values.get("name", ""), "value": values.get("value", ""), "form": values.get("form", ""), "id": values.get("id", ""), "class": values.get("class", ""), "text_preview": preview(match.group("body") or "")})
    return rows
