"""CSV export for collapsed Markdown sections."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "section_type", "label", "starts_open"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DETAILS_RE = re.compile(r"<details\b([^>]*)>", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary\s*>", re.IGNORECASE)
_OPEN_RE = re.compile(r"(?:^|\s)open(?:\s|=|$)", re.IGNORECASE)
_CALLOUT_RE = re.compile(r"^\s*>\s*\[![^\]]+\]-\s*(.*)$")
_TAG_RE = re.compile(r"<[^>]+>")


def export_units_to_markdown_collapsed_section_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["section_type"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int | bool]] = []
    pending_details: dict[str, str | int | bool] | None = None
    for line_number, line in _content_lines(str(get(unit, "content") or "")):
        details = _DETAILS_RE.search(line)
        if details:
            label = _summary(line)
            row = {"unit_id": uid, "title": title, "line_number": line_number, "section_type": "details", "label": label, "starts_open": bool(_OPEN_RE.search(details.group(1)))}
            if label:
                rows.append(row)
            else:
                pending_details = row
            continue
        if pending_details is not None:
            label = _summary(line)
            if label:
                pending_details["label"] = label
                rows.append(pending_details)
                pending_details = None
        callout = _CALLOUT_RE.match(line)
        if callout:
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "section_type": "obsidian_callout", "label": field_value(callout.group(1)), "starts_open": False})
    if pending_details is not None:
        rows.append(pending_details)
    return rows


def _summary(line: str) -> str:
    match = _SUMMARY_RE.search(line)
    return field_value(" ".join(_TAG_RE.sub(" ", match.group(1)).split())) if match else ""


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
