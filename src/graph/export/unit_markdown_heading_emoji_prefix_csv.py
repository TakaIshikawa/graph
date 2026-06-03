"""CSV export for Markdown headings that start with an emoji."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "heading_depth", "emoji", "heading_text"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ATX_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^\s*(=+|-+)\s*$")
_EMOJI_RE = re.compile(r"^([\U0001F000-\U0001FAFF\u2600-\u27BF](?:\ufe0f|\u200d[\U0001F000-\U0001FAFF\u2600-\u27BF])*)\s*(.*)$")


def export_unit_markdown_heading_emoji_prefix_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["heading_depth"]), sort_key(row["heading_text"])))
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
    previous_text = ""
    previous_line = 0
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            previous_text = ""
            previous_line = 0
            continue
        if in_fence:
            continue
        atx = _ATX_RE.match(line)
        if atx:
            row = _heading_row(uid, title, source, line_number, len(atx.group(1)), atx.group(2))
            if row:
                rows.append(row)
            previous_text = ""
            previous_line = 0
            continue
        setext = _SETEXT_RE.match(line)
        if setext and previous_text:
            row = _heading_row(uid, title, source, previous_line, 1 if setext.group(1).startswith("=") else 2, previous_text)
            if row:
                rows.append(row)
            previous_text = ""
            previous_line = 0
            continue
        previous_text = field_value(line) if line.strip() else ""
        previous_line = line_number if previous_text else 0
    return rows


def _heading_row(uid: str, title: str, source: str, line_number: int, depth: int, text: str) -> dict[str, str | int] | None:
    match = _EMOJI_RE.match(field_value(text))
    if not match:
        return None
    return {"unit_id": uid, "title": title, "source": source, "line_number": line_number, "heading_depth": depth, "emoji": field_value(match.group(1)), "heading_text": field_value(match.group(2))}
