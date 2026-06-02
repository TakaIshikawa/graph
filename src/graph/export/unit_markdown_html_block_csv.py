"""CSV export for block-level HTML embedded in Markdown."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "tag_name", "raw_attributes", "class_attribute", "id_value", "self_contained"]
_TAGS = "div|section|article|aside|blockquote|pre|table"
_OPEN_RE = re.compile(rf"^[ \t]{{0,3}}<(?P<tag>{_TAGS})\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"\b(?P<name>class|id)\s*=\s*(['\"])(?P<value>.*?)\2", re.IGNORECASE)
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def export_units_to_markdown_html_block_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag_name"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    rows: list[dict[str, str | int | bool]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _OPEN_RE.match(line)
        if not match:
            continue
        attrs = {m.group("name").casefold(): field_value(m.group("value")) for m in _ATTR_RE.finditer(match.group("attrs"))}
        tag = match.group("tag").casefold()
        rows.append({"unit_id": uid, "title": title, "source": source, "line_number": line_number, "tag_name": tag, "raw_attributes": field_value(match.group("attrs")), "class_attribute": attrs.get("class", ""), "id_value": attrs.get("id", ""), "self_contained": bool(re.search(rf"</{tag}\s*>", line[match.end() :], re.IGNORECASE))})
    return rows
