"""CSV export for HTML footnote backlink anchors in Markdown output."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "footnote_id", "href", "backref_text"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ANCHOR_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<text>.*?)</a>", re.IGNORECASE)
_HREF_RE = re.compile(r"""href\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""", re.IGNORECASE)


def export_unit_markdown_footnote_backref_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
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
        for anchor in _ANCHOR_RE.finditer(line):
            href_match = _HREF_RE.search(anchor.group("attrs"))
            href = next((value for value in href_match.groups() if value is not None), "") if href_match else ""
            if not href.startswith("#fnref"):
                continue
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "footnote_id": href[6:], "href": href, "backref_text": field_value(re.sub(r"<[^>]+>", "", anchor.group("text")))})
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
