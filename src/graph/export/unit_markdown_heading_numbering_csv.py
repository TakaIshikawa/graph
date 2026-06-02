"""CSV export for numbered Markdown headings."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "heading_depth", "numbering_style", "raw_number", "normalized_number", "heading_text"]
_ATX_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_SETEXT_RE = re.compile(r"^[ \t]*(=+|-+)[ \t]*$")
_NUM_RE = re.compile(r"^(?P<num>(?:\d+[.)]?|\d+(?:\.\d+)+\.?|[A-Za-z][.)]|[IVXLCDMivxlcdm]+\.))[ \t]+(?P<text>.+)$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def export_units_to_markdown_heading_numbering_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
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
    lines = str(get(unit, "content") or data.get("content") or "").splitlines()
    rows: list[dict[str, str | int]] = []
    in_fence = False
    previous: tuple[int, str] | None = None
    for line_number, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            previous = None
            continue
        if in_fence:
            continue
        atx = _ATX_RE.match(line)
        if atx:
            row = _numbered_row(uid, title, source, line_number, len(atx.group(1)), atx.group(2))
            if row:
                rows.append(row)
            previous = None
            continue
        setext = _SETEXT_RE.match(line)
        if setext and previous:
            row = _numbered_row(uid, title, source, previous[0], 1 if setext.group(1).startswith("=") else 2, previous[1])
            if row:
                rows.append(row)
            previous = None
            continue
        previous = (line_number, line.strip()) if line.strip() and not line.startswith((" ", "\t")) else None
    return rows


def _numbered_row(uid: str, title: str, source: str, line_number: int, depth: int, text: str) -> dict[str, str | int] | None:
    match = _NUM_RE.match(field_value(text))
    if not match:
        return None
    raw = match.group("num")
    normalized = raw.rstrip(".)").casefold()
    return {
        "unit_id": uid,
        "title": title,
        "source": source,
        "line_number": line_number,
        "heading_depth": depth,
        "numbering_style": _style(raw),
        "raw_number": raw,
        "normalized_number": normalized,
        "heading_text": field_value(match.group("text")),
    }


def _style(raw: str) -> str:
    token = raw.rstrip(".)")
    if re.fullmatch(r"\d+(?:\.\d+)+", token):
        return "dotted"
    if token.isdigit():
        return "numeric"
    if re.fullmatch(r"[IVXLCDMivxlcdm]+", token) and len(token) > 1:
        return "roman"
    return "alphabetic"
