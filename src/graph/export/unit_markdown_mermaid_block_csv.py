"""CSV export for Markdown Mermaid fenced blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "start_line", "end_line", "diagram_type", "line_count"]
_OPEN_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


def export_units_to_markdown_mermaid_block_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["start_line"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    lines = str(get(unit, "content") or metadata(unit).get("content") or "").splitlines()
    active: dict[str, Any] | None = None
    body: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if active:
            if _is_closing(line, active["char"], active["length"]):
                rows.append(_row(unit, title, active, body, line_number))
                active = None
                body = []
            else:
                body.append(line)
            continue
        match = _OPEN_RE.match(line)
        if match and _language(match.group("info")) == "mermaid":
            active = {"start_line": line_number, "char": match.group("fence")[0], "length": len(match.group("fence"))}
    if active:
        rows.append(_row(unit, title, active, body, len(lines)))
    return rows


def _row(unit: Mapping[str, Any] | object, title: str, active: dict[str, Any], body: list[str], end_line: int) -> dict[str, str | int]:
    first = next((field_value(line) for line in body if field_value(line)), "")
    return {
        "unit_id": unit_id(unit),
        "title": title,
        "start_line": active["start_line"],
        "end_line": end_line,
        "diagram_type": first.split()[0].casefold() if first else "",
        "line_count": len(body),
    }


def _language(info: str) -> str:
    return field_value(info).split(None, 1)[0].casefold() if field_value(info) else ""


def _is_closing(line: str, char: str, length: int) -> bool:
    stripped = line.strip()
    return stripped.startswith(char * length) and set(stripped) == {char}
