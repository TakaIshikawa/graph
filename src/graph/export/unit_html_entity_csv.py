"""CSV export for HTML entities in unit content."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "entity", "entity_type", "decoded"]
_ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]+|#[0-9]+|#[xX][0-9A-Fa-f]+);")


def export_units_to_html_entity_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["entity"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows = []
    in_fence = False
    content = get(unit, "content")
    for line_number, line in enumerate(("" if content is None else str(content)).splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _ENTITY_RE.finditer(line):
            entity = match.group(0)
            rows.append({"unit_id": unit_id(unit), "title": field_value(get(unit, "title")), "line_number": line_number, "entity": entity, "entity_type": _entity_type(entity), "decoded": html.unescape(entity)})
    return rows


def _entity_type(entity: str) -> str:
    if entity.startswith("&#x") or entity.startswith("&#X"):
        return "hex"
    if entity.startswith("&#"):
        return "decimal"
    return "named"
