"""CSV export for inline HTML style attributes in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "tag_name", "style_text", "property_count", "raw_tag"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)(?:\s[^<>]*)?/?>")
_STYLE_RE = re.compile(r"""(?:^|\s)style\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def export_unit_markdown_html_style_attribute_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag_name"]), sort_key(row["style_text"])))
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
        clean_line = _COMMENT_RE.sub("", line)
        for tag_match in _TAG_RE.finditer(clean_line):
            raw_tag = tag_match.group(0)
            style_match = _STYLE_RE.search(raw_tag)
            if not style_match:
                continue
            style_text = field_value(next(group for group in style_match.groups() if group is not None))
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "line_number": line_number,
                    "tag_name": tag_match.group(1).casefold(),
                    "style_text": style_text,
                    "property_count": _property_count(style_text),
                    "raw_tag": raw_tag,
                }
            )
    return rows


def _property_count(style_text: str) -> int:
    return sum(1 for part in style_text.split(";") if part.strip())
