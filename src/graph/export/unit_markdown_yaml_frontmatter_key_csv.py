"""CSV export for top-level YAML frontmatter keys in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "key", "has_value"]
_KEY_RE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+)\s*:\s*(?P<value>.*)$")


def export_unit_markdown_yaml_frontmatter_keys_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["key"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    lines = str(get(unit, "content") or metadata(unit).get("content") or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    end_index = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end_index is None:
        return []
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    body = lines[1:end_index]
    for offset, line in enumerate(body, start=2):
        if line[:1].isspace() or line.strip().startswith("#"):
            continue
        match = _KEY_RE.match(line)
        if not match:
            continue
        rows.append({"unit_id": uid, "title": title, "line_number": offset, "key": match.group("key"), "has_value": "true" if _has_value(match.group("value"), body[offset - 1 :]) else "false"})
    return rows


def _has_value(value: str, following_lines: list[str]) -> bool:
    if field_value(value):
        return True
    for line in following_lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[:1].isspace():
            return False
        return True
    return False
