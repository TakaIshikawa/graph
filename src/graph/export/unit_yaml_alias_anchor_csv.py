"""CSV inventory for YAML frontmatter anchors and aliases."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "symbol_type", "symbol_name", "key_path", "raw_value"]
_SYMBOL_RE = re.compile(r"(?<!\w)([&*])([A-Za-z0-9_-]+)")


def export_units_to_yaml_alias_anchor_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write YAML anchors and aliases from leading frontmatter only."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row} for row in _symbol_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["symbol_type"]), sort_key(row["symbol_name"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _symbol_rows(content: str) -> list[dict[str, str | int]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    rows: list[dict[str, str | int]] = []
    path_stack: list[tuple[int, str]] = []
    for offset, line in enumerate(lines[1:], start=2):
        if line.strip() in {"---", "..."}:
            return rows
        key_path = _key_path(line, path_stack)
        raw_value = field_value(line)
        for match in _SYMBOL_RE.finditer(line):
            rows.append({"line_number": offset, "symbol_type": "anchor" if match.group(1) == "&" else "alias", "symbol_name": match.group(2), "key_path": key_path, "raw_value": raw_value})
    return []


def _key_path(line: str, stack: list[tuple[int, str]]) -> str:
    match = re.match(r"^(\s*)(?:-\s*)?([A-Za-z0-9_.-]+)\s*:", line)
    if not match:
        return ".".join(key for _, key in stack)
    indent = len(match.group(1))
    key = match.group(2)
    while stack and stack[-1][0] >= indent:
        stack.pop()
    stack.append((indent, key))
    return ".".join(item for _, item in stack)
