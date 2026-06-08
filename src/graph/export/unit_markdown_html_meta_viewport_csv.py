"""CSV export for Markdown-embedded HTML meta viewport elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "content", "width_value", "initial_scale", "maximum_scale", "user_scalable", "disables_zoom", "id", "class"]
_META_RE = re.compile(r"<meta\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_meta_viewport_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["content"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _META_RE.finditer(content):
        values = attrs(match.group("attrs"))
        if values.get("name", "").casefold() != "viewport":
            continue
        parsed = _directives(values.get("content", ""))
        max_scale = parsed.get("maximum-scale", "")
        user_scalable = parsed.get("user-scalable", "")
        disables_zoom = user_scalable.casefold() in {"no", "0", "false"} or _scale_blocks_zoom(max_scale)
        rows.append({**context, "line_number": line_number(content, match.start()), "content": values.get("content", ""), "width_value": parsed.get("width", ""), "initial_scale": parsed.get("initial-scale", ""), "maximum_scale": max_scale, "user_scalable": user_scalable, "disables_zoom": str(disables_zoom).lower(), "id": values.get("id", ""), "class": values.get("class", "")})
    return rows


def _directives(content: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in content.split(","):
        key, _, value = part.strip().partition("=")
        if key:
            result[key.casefold()] = value.strip()
    return result


def _scale_blocks_zoom(value: str) -> bool:
    try:
        return bool(value) and float(value) <= 1
    except ValueError:
        return False
