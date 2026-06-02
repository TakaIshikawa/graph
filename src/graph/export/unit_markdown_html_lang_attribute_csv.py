"""CSV export for HTML lang attributes in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "tag_name", "lang", "raw_snippet"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)([^<>]*)>")
_LANG_RE = re.compile(r"""\slang\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)


def export_unit_markdown_html_lang_attributes_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag_name"]), sort_key(row["lang"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in _content_lines(str(get(unit, "content") or "")):
        for tag_match in _TAG_RE.finditer(line):
            attr_match = _LANG_RE.search(tag_match.group(2))
            if attr_match:
                rows.append(
                    {
                        "unit_id": uid,
                        "title": title,
                        "line_number": line_number,
                        "tag_name": tag_match.group(1).casefold(),
                        "lang": field_value(next((part for part in attr_match.groups() if part is not None), "")).casefold(),
                        "raw_snippet": field_value(tag_match.group(0)),
                    }
                )
    return rows


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
