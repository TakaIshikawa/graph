"""CSV export for Markdown-embedded HTML picture source elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "tag", "src", "srcset", "media", "type", "sizes", "alt", "raw_html"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_PICTURE_OPEN_RE = re.compile(r"<picture\b[^>]*>", re.IGNORECASE)
_PICTURE_CLOSE_RE = re.compile(r"</picture\s*>", re.IGNORECASE)
_IMAGE_TAG_RE = re.compile(r"<(?P<tag>source|img)\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""")


def export_unit_markdown_html_picture_source_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(
        key=lambda row: (
            sort_key(row["unit_id"]),
            int(row["line_number"]),
            sort_key(row["tag"]),
            sort_key(row["srcset"]),
            sort_key(row["src"]),
        )
    )
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    start_line = 0
    block: list[str] = []

    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not block:
            match = _PICTURE_OPEN_RE.search(line)
            if not match:
                continue
            start_line = line_number
            block = [line[match.start() :]]
        else:
            block.append(line)
        if _PICTURE_CLOSE_RE.search(line):
            rows.extend(_picture_rows(uid, title, start_line, "\n".join(block)))
            block = []

    if block:
        rows.extend(_picture_rows(uid, title, start_line, "\n".join(block)))
    return rows


def _picture_rows(uid: str, title: str, line_number: int, block: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for match in _IMAGE_TAG_RE.finditer(block):
        tag = match.group("tag").lower()
        attrs = _attrs(match.group("attrs"))
        rows.append(
            {
                "unit_id": uid,
                "title": title,
                "line_number": line_number,
                "tag": tag,
                "src": attrs.get("src", ""),
                "srcset": attrs.get("srcset", ""),
                "media": attrs.get("media", ""),
                "type": attrs.get("type", ""),
                "sizes": attrs.get("sizes", ""),
                "alt": attrs.get("alt", ""),
                "raw_html": match.group(0),
            }
        )
    return rows


def _attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw):
        attrs[match.group(1).casefold()] = field_value(match.group(2) or match.group(3) or match.group(4))
    return attrs
