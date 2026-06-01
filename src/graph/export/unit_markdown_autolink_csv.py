"""CSV inventory for Markdown angle autolinks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "target", "target_type", "line_number", "source_url"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ANGLE_RE = re.compile(r"<([^<>\s]+)>")
_EMAIL_RE = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")


def export_units_to_markdown_autolink_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per URL or email angle autolink outside fenced code."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        source_url = field_value(get(unit, "source_url") or metadata(unit).get("source_url") or metadata(unit).get("url"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row, "source_url": source_url} for row in _autolink_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _autolink_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _ANGLE_RE.finditer(line):
            target = match.group(1)
            kind = _kind(target)
            if kind:
                rows.append({"target": target, "target_type": kind, "line_number": line_number})
    return rows


def _kind(target: str) -> str:
    parsed = urlparse(target)
    if parsed.scheme and (parsed.netloc or parsed.scheme.casefold() == "mailto"):
        return "email" if parsed.scheme.casefold() == "mailto" else "url"
    return "email" if _EMAIL_RE.match(target) else ""
