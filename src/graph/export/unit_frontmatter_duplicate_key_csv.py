"""CSV export for duplicate keys in leading YAML frontmatter."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "key", "first_line_number", "duplicate_line_number", "occurrence_count"]
_KEY_RE = re.compile(r"^\s*([^\s:#][^:#]*?)\s*:")


def export_unit_frontmatter_duplicate_key_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["duplicate_line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    seen: dict[str, tuple[str, int, int]] = {}
    rows: list[dict[str, str | int]] = []
    for line_number, line in _frontmatter_lines(str(get(unit, "content") or "")):
        match = _KEY_RE.match(line)
        if not match or line.lstrip().startswith("-"):
            continue
        key = match.group(1).strip()
        normalized = key.casefold()
        if normalized not in seen:
            seen[normalized] = (key, line_number, 1)
            continue
        first_key, first_line, count = seen[normalized]
        count += 1
        seen[normalized] = (first_key, first_line, count)
        rows.append(
            {
                "unit_id": uid,
                "title": title,
                "key": key,
                "first_line_number": first_line,
                "duplicate_line_number": line_number,
                "occurrence_count": count,
            }
        )
    return rows


def _frontmatter_lines(content: str) -> list[tuple[int, str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    rows: list[tuple[int, str]] = []
    for offset, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            return rows
        rows.append((offset, line))
    return []
