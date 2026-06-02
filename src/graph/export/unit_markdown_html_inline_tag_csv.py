"""CSV export for inline HTML tags in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "tag_name", "closing", "raw_tag"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9:-]*)(?:\s[^<>]*)?/?>")
_BLOCK_LINE_RE = re.compile(r"^\s{0,3}</?([A-Za-z][A-Za-z0-9:-]*)(?:\s|>|/>)\s*$")


def export_unit_markdown_html_inline_tags_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or "<!--" in line or _BLOCK_LINE_RE.match(line):
            continue
        for match in _TAG_RE.finditer(line):
            raw_tag = match.group(0)
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "tag_name": match.group(1).casefold(), "closing": "true" if raw_tag.startswith("</") else "false", "raw_tag": raw_tag})
    return rows
