"""CSV export for Markdown-embedded HTML form validation attributes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "type", "name", "required", "pattern", "minlength", "maxlength", "min", "max", "step", "novalidate", "id", "class"]
_TAG_RE = re.compile(r"<(?P<tag>form|input|textarea|select)\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_form_validation_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["name"])))
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
        values = attrs(match.group("attrs"))
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": match.group("tag").casefold(), "type": values.get("type", ""), "name": values.get("name", ""), "required": bool_attr(values, "required"), "pattern": values.get("pattern", ""), "minlength": values.get("minlength", ""), "maxlength": values.get("maxlength", ""), "min": values.get("min", ""), "max": values.get("max", ""), "step": values.get("step", ""), "novalidate": bool_attr(values, "novalidate"), "id": values.get("id", ""), "class": values.get("class", "")})
    return rows
