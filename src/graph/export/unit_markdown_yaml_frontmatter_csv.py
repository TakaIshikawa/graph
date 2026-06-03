"""CSV export for YAML frontmatter key paths in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "key_path", "value_excerpt", "value_kind", "line_number"]
_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_.-]+)\s*:\s*(?P<value>.*)$")
_LIST_RE = re.compile(r"^(?P<indent>\s*)-\s*(?P<value>.*)$")


def export_units_to_markdown_yaml_frontmatter_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["key_path"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    lines = str(get(unit, "content") or metadata(unit).get("content") or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    end = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        return []
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    stack: list[tuple[int, str]] = []
    for offset, line in enumerate(lines[1:end], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if match := _KEY_RE.match(line):
            indent = len(match.group("indent"))
            stack = [(level, key) for level, key in stack if level < indent]
            key_path = ".".join([key for _, key in stack] + [match.group("key")])
            value = field_value(match.group("value"))
            kind = _value_kind(value)
            rows.append({"unit_id": uid, "title": title, "key_path": key_path, "value_excerpt": value[:120], "value_kind": kind, "line_number": offset})
            if not value:
                stack.append((indent, match.group("key")))
        elif match := _LIST_RE.match(line):
            parent = ".".join(key for _, key in stack)
            if parent:
                value = field_value(match.group("value"))
                rows.append({"unit_id": uid, "title": title, "key_path": f"{parent}[]", "value_excerpt": value[:120], "value_kind": "list_item", "line_number": offset})
    return rows


def _value_kind(value: str) -> str:
    if not value:
        return "mapping_or_list"
    if value.startswith("[") and value.endswith("]"):
        return "list"
    if value.startswith("{") and value.endswith("}"):
        return "mapping"
    return "scalar"
