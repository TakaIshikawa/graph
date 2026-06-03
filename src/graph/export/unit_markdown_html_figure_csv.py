"""CSV export for HTML figure blocks embedded in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "has_caption", "caption_text", "image_sources"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_OPEN_RE = re.compile(r"<figure\b[^>]*>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</figure\s*>", re.IGNORECASE)
_CAPTION_RE = re.compile(r"<figcaption\b[^>]*>(?P<text>.*?)</figcaption\s*>", re.IGNORECASE | re.DOTALL)
_IMG_RE = re.compile(r"<img\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_SRC_RE = re.compile(r"""(?:^|\s)src\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def export_unit_markdown_html_figure_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
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
    start_line = 0
    block: list[str] = []
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not block:
            match = _OPEN_RE.search(line)
            if not match:
                continue
            start_line = line_number
            block = [line[match.start() :]]
        else:
            block.append(line)
        if _CLOSE_RE.search(line):
            rows.append(_row(uid, title, source, start_line, "\n".join(block)))
            block = []
    if block:
        rows.append(_row(uid, title, source, start_line, "\n".join(block)))
    return rows


def _row(uid: str, title: str, source: str, line_number: int, block: str) -> dict[str, str | int | bool]:
    caption = _CAPTION_RE.search(block)
    image_sources = []
    for image in _IMG_RE.finditer(block):
        source_match = _SRC_RE.search(image.group("attrs"))
        if source_match:
            image_sources.append(field_value(source_match.group(1) or source_match.group(2) or source_match.group(3)))
    return {
        "unit_id": uid,
        "title": title,
        "source": source,
        "line_number": line_number,
        "has_caption": bool(caption),
        "caption_text": field_value(_TAG_RE.sub(" ", caption.group("text"))) if caption else "",
        "image_sources": "; ".join(image_sources),
    }
