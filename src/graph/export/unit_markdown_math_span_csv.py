"""CSV export for Markdown inline math spans."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "expression", "character_length", "delimiter_style", "position"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def export_units_to_markdown_math_span_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["position"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or "$$" in line:
            continue
        for start, end, expr in _math_spans(line):
            rows.append({
                "unit_id": uid,
                "title": title,
                "source": source,
                "line_number": line_number,
                "expression": field_value(expr),
                "character_length": len(expr),
                "delimiter_style": "$",
                "position": start + 1,
            })
    return rows


def _math_spans(line: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(line):
        start = line.find("$", index)
        if start < 0:
            break
        if _escaped(line, start) or _inside_code_span(line, start) or (start + 1 < len(line) and line[start + 1].isdigit()):
            index = start + 1
            continue
        end = line.find("$", start + 1)
        while end >= 0 and _escaped(line, end):
            end = line.find("$", end + 1)
        if end < 0:
            break
        expr = line[start + 1 : end]
        if expr.strip() and not expr[0].isspace() and not expr[-1].isspace():
            spans.append((start, end + 1, expr))
        index = end + 1
    return spans


def _escaped(line: str, offset: int) -> bool:
    return (len(line[:offset]) - len(line[:offset].rstrip("\\"))) % 2 == 1


def _inside_code_span(line: str, offset: int) -> bool:
    return line[:offset].count("`") % 2 == 1
