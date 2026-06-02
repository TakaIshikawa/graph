"""CSV export for Markdown links with empty visible text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "link_type", "target", "raw_link"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_RE = re.compile(r"(?<!!)\[(?P<label>[^\]\n]*)\]\((?P<target>[^)\n]*)\)")
_REFERENCE_RE = re.compile(r"(?<!!)\[(?P<label>[^\]\n]*)\]\[(?P<target>[^\]\n]*)\]")


def export_unit_markdown_empty_link_texts_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["link_type"]), sort_key(row["target"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        inline_spans: list[tuple[int, int]] = []
        for match in _INLINE_RE.finditer(line):
            inline_spans.append(match.span())
            if not field_value(match.group("label")):
                rows.append(_row(uid, title, line_number, "inline", match.group("target"), match.group(0)))
        for match in _REFERENCE_RE.finditer(line):
            if any(start <= match.start() < end for start, end in inline_spans):
                continue
            if not field_value(match.group("label")):
                rows.append(_row(uid, title, line_number, "reference", match.group("target"), match.group(0)))
    return rows


def _row(uid: str, title: str, line_number: int, link_type: str, target: str, raw_link: str) -> dict[str, str | int]:
    return {"unit_id": uid, "title": title, "line_number": line_number, "link_type": link_type, "target": field_value(target), "raw_link": raw_link}
