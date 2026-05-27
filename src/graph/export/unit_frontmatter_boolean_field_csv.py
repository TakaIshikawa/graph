"""CSV export for boolean YAML frontmatter fields."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "key_path", "value", "original_token", "line_number"]
_PAIR_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^:#][^:]*):\s*(?P<value>[^#]*?)(?:\s+#.*)?$")


def export_units_to_frontmatter_boolean_field_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["key_path"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    block, start_line = _frontmatter_block(str(get(unit, "content") or ""))
    if not block:
        return []
    try:
        parsed = yaml.safe_load(block) or {}
    except yaml.YAMLError:
        return []
    if not isinstance(parsed, Mapping):
        return []
    tokens = _scalar_tokens(block, start_line)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for key_path, value in _flatten(parsed):
        if isinstance(value, bool):
            token, line = tokens.get(key_path, ("", ""))
            rows.append({"unit_id": uid, "title": title, "key_path": key_path, "value": str(value).lower(), "original_token": token, "line_number": line})
    return rows


def _frontmatter_block(content: str) -> tuple[str, int]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", 0
    body: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return "\n".join(body), 2
        body.append(line)
    return "", 0


def _flatten(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    for key, child in value.items():
        path = f"{prefix}.{field_value(key)}" if prefix else field_value(key)
        if isinstance(child, Mapping):
            rows.extend(_flatten(child, path))
        else:
            rows.append((path, child))
    return rows


def _scalar_tokens(block: str, start_line: int) -> dict[str, tuple[str, int]]:
    tokens: dict[str, tuple[str, int]] = {}
    stack: list[tuple[int, str]] = []
    for offset, line in enumerate(block.splitlines(), start=start_line):
        match = _PAIR_RE.match(line)
        if not match:
            continue
        indent = len(match.group("indent"))
        key = field_value(match.group("key").strip().strip("'\""))
        value = match.group("value").strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        path = ".".join([item[1] for item in stack] + [key])
        if value:
            tokens[path] = (value.strip("'\""), offset)
        else:
            stack.append((indent, key))
    return tokens
