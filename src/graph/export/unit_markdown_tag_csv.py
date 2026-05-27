"""CSV export for inline Markdown hashtag tags."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "tag", "normalized_tag", "depth", "line_number", "context"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_TAG_RE = re.compile(r"(?<![\w/&?=])#([A-Za-z][A-Za-z0-9_-]*(?:/[A-Za-z0-9_-]+)*)")
_CODE_RE = re.compile(r"`+[^`\n]*`+")


def export_units_to_markdown_tag_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["normalized_tag"])))
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
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _HEADING_RE.match(line):
            continue
        searchable = _CODE_RE.sub("", line)
        for match in _TAG_RE.finditer(searchable):
            tag = f"#{match.group(1)}"
            normalized = tag.casefold()
            rows.append({"unit_id": uid, "title": title, "tag": tag, "normalized_tag": normalized, "depth": normalized.count("/") + 1, "line_number": line_number, "context": field_value(line)})
    return rows
