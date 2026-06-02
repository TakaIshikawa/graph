"""CSV export for inline HTML kbd tags in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import WHITESPACE_RE, field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "key_text", "raw_html"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_KBD_RE = re.compile(r"<kbd\b[^>]*>(.*?)</kbd\s*>", re.IGNORECASE)


def export_unit_markdown_kbd_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["key_text"]), sort_key(row["raw_html"])))
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
        for match in _KBD_RE.finditer(line):
            key_text = WHITESPACE_RE.sub(" ", field_value(match.group(1))).strip()
            if key_text:
                rows.append({"unit_id": uid, "title": title, "line_number": line_number, "key_text": key_text, "raw_html": match.group(0)})
    return rows
