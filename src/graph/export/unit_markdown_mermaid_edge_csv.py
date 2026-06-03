"""CSV export for graph edges in Markdown Mermaid blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "diagram_type", "from_node", "arrow", "to_node", "edge_label"]
_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*mermaid\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_EDGE_RE = re.compile(r"^\s*(?P<from>.+?)\s*(?P<arrow>-\.-?>|==>|-->|---)\s*(?:\|(?P<label>[^|]*)\|\s*)?(?P<to>.+?)\s*;?\s*$")


def export_units_to_markdown_mermaid_edge_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["from_node"]), sort_key(row["to_node"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    meta = metadata(unit)
    title = field_value(get(unit, "title") or meta.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_id") or meta.get("source") or meta.get("source_id"))
    rows: list[dict[str, str | int]] = []
    in_mermaid = False
    diagram_type = "unknown"
    for line_number, line in enumerate(str(get(unit, "content") or meta.get("content") or "").splitlines(), start=1):
        if in_mermaid:
            if _FENCE_RE.match(line):
                in_mermaid = False
                diagram_type = "unknown"
                continue
            statement = field_value(line)
            if not statement:
                continue
            if diagram_type == "unknown":
                diagram_type = statement.split()[0]
            edge = _edge(statement)
            if edge:
                rows.append({"unit_id": uid, "title": title, "source": source, "line_number": line_number, "diagram_type": diagram_type, **edge})
            continue
        if _OPEN_RE.match(line):
            in_mermaid = True
    return rows


def _edge(statement: str) -> dict[str, str] | None:
    text = statement.split("%%", 1)[0].strip()
    match = _EDGE_RE.match(text)
    if not match:
        return None
    return {
        "from_node": field_value(match.group("from")),
        "arrow": match.group("arrow"),
        "to_node": field_value(match.group("to")),
        "edge_label": field_value(match.group("label")),
    }
