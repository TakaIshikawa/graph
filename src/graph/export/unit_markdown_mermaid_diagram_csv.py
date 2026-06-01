"""CSV inventory for Mermaid fenced code blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "diagram_type", "statement_count", "first_statement"]
_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*mermaid\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_markdown_mermaid_diagram_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per Mermaid fenced code block."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _diagram_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["first_statement"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _diagram_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_mermaid = False
    opening_line = 0
    statements: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if in_mermaid:
            if _FENCE_RE.match(line):
                rows.append(_row(opening_line, statements))
                in_mermaid = False
                statements = []
            else:
                statement = field_value(line)
                if statement:
                    statements.append(statement)
            continue
        if _OPEN_RE.match(line):
            in_mermaid = True
            opening_line = line_number
    if in_mermaid:
        rows.append(_row(opening_line, statements))
    return rows


def _row(opening_line: int, statements: list[str]) -> dict[str, str | int]:
    first = statements[0] if statements else ""
    return {
        "line_number": opening_line,
        "diagram_type": first.split()[0] if first else "unknown",
        "statement_count": len(statements),
        "first_statement": first,
    }
