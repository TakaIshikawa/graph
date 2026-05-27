"""CSV export for HTML details blocks in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "start_line", "end_line", "summary", "is_open", "line_count"]
_DETAILS_OPEN_RE = re.compile(r"<details\b([^>]*)>", re.IGNORECASE)
_DETAILS_CLOSE_RE = re.compile(r"</details\s*>", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary\s*>", re.IGNORECASE)
_OPEN_ATTR_RE = re.compile(r"(?:^|\s)open(?:\s|=|$)", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_markdown_details_inventory_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["start_line"]), sort_key(row["summary"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    lines = list(_content_lines(str(get(unit, "content") or "")))
    rows: list[dict[str, str | int]] = []
    active: dict[str, Any] | None = None
    for line_number, line in lines:
        if active is None:
            match = _DETAILS_OPEN_RE.search(line)
            if not match:
                continue
            active = {"start": line_number, "summary": "", "is_open": bool(_OPEN_ATTR_RE.search(match.group(1)))}
        summary = _SUMMARY_RE.search(line)
        if summary and not active["summary"]:
            active["summary"] = field_value(summary.group(1))
        if _DETAILS_CLOSE_RE.search(line):
            rows.append(_row(uid, title, active, line_number))
            active = None
    if active is not None:
        rows.append(_row(uid, title, active, lines[-1][0] if lines else active["start"]))
    return rows


def _row(uid: str, title: str, active: dict[str, Any], end_line: int) -> dict[str, str | int]:
    return {
        "unit_id": uid,
        "title": title,
        "start_line": active["start"],
        "end_line": end_line,
        "summary": active["summary"],
        "is_open": "true" if active["is_open"] else "false",
        "line_count": end_line - active["start"] + 1,
    }


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
