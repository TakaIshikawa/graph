"""CSV export for Markdown frontmatter scalar keys."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "key_path", "value_preview", "value_type", "line_number"]


def export_units_to_markdown_frontmatter_key_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["key_path"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    lines = str(get(unit, "content") or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    closing = next((index for index, line in enumerate(lines[1:], start=2) if line.strip() == "---"), 0)
    if not closing:
        return []
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_project") or metadata(unit).get("source"))
    stack: list[tuple[int, str]] = []
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(lines[1 : closing - 1], start=2):
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, raw_value = line.strip().split(":", 1)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key = field_value(key)
        value = raw_value.strip()
        path = ".".join([item[1] for item in stack] + [key])
        if value == "":
            stack.append((indent, key))
            continue
        rows.append(
            {
                "unit_id": unit_id(unit),
                "title": title,
                "source": source,
                "key_path": path,
                "value_preview": field_value(value)[:120],
                "value_type": _value_type(value),
                "line_number": line_number,
            }
        )
    return rows


def _value_type(value: str) -> str:
    text = value.strip().strip("'\"")
    if value.strip() in {"[]", "{}"}:
        return "collection"
    if text.casefold() in {"true", "false"}:
        return "boolean"
    if text.casefold() in {"null", "~"}:
        return "null"
    try:
        float(text)
    except ValueError:
        return "string"
    return "number"
