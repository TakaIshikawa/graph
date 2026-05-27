"""CSV export for Mermaid fenced code blocks by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "diagram_type", "line_count", "starts_at_line"]
_FENCE_RE = re.compile(r"^\s*(```|~~~)\s*mermaid\b", re.IGNORECASE)


def export_units_to_mermaid_diagram_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _diagrams(str(get(unit, "content") or "")))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["starts_at_line"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _diagrams(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    lines = content.splitlines()
    index = 0
    while index < len(lines):
        start = _FENCE_RE.match(lines[index])
        if not start:
            index += 1
            continue
        fence = start.group(1)
        starts_at = index + 1
        body: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].lstrip().startswith(fence):
            body.append(lines[index])
            index += 1
        rows.append({"diagram_type": _diagram_type(body), "line_count": len(body), "starts_at_line": starts_at})
        index += 1
    return rows


def _diagram_type(lines: list[str]) -> str:
    first = next((field_value(line) for line in lines if field_value(line)), "")
    lower = first.casefold()
    for kind in ("flowchart", "sequencediagram", "classdiagram", "graph", "gantt"):
        if lower.startswith(kind):
            return {"sequencediagram": "sequenceDiagram", "classdiagram": "classDiagram"}.get(kind, kind)
    return "unknown"
