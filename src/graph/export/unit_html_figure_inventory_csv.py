"""CSV export for HTML figure blocks in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "has_figcaption", "figcaption", "image_count", "link_count"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_FIGURE_OPEN_RE = re.compile(r"<figure\b[^>]*>", re.IGNORECASE)
_FIGURE_CLOSE_RE = re.compile(r"</figure\s*>", re.IGNORECASE)
_FIGCAPTION_RE = re.compile(r"<figcaption\b[^>]*>(.*?)</figcaption\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_IMG_RE = re.compile(r"<img\b", re.IGNORECASE)
_LINK_RE = re.compile(r"<a\b", re.IGNORECASE)


def export_units_to_html_figure_inventory_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    lines = _content_lines(str(get(unit, "content") or ""))
    rows: list[dict[str, str | int | bool]] = []
    active: dict[str, Any] | None = None
    for line_number, line in lines:
        if active is None and _FIGURE_OPEN_RE.search(line):
            active = {"line_number": line_number, "parts": []}
        if active is None:
            continue
        active["parts"].append(line)
        if _FIGURE_CLOSE_RE.search(line):
            body = "\n".join(active["parts"])
            caption_match = _FIGCAPTION_RE.search(body)
            caption = _clean(caption_match.group(1)) if caption_match else ""
            rows.append({"unit_id": uid, "title": title, "line_number": active["line_number"], "has_figcaption": bool(caption_match), "figcaption": caption, "image_count": len(_IMG_RE.findall(body)), "link_count": len(_LINK_RE.findall(body))})
            active = None
    if active is not None:
        body = "\n".join(active["parts"])
        caption_match = _FIGCAPTION_RE.search(body)
        caption = _clean(caption_match.group(1)) if caption_match else ""
        rows.append({"unit_id": uid, "title": title, "line_number": active["line_number"], "has_figcaption": bool(caption_match), "figcaption": caption, "image_count": len(_IMG_RE.findall(body)), "link_count": len(_LINK_RE.findall(body))})
    return rows


def _clean(text: str) -> str:
    return field_value(" ".join(_TAG_RE.sub(" ", text).split()))


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
