"""CSV export for Markdown image dimension hints."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "alt_text", "target", "width", "height", "hint_style", "line_number", "context"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_SIZE_RE = re.compile(r"(?:^|\s)=([0-9]+)?x([0-9]+)?(?:\s|$)")
_PIPE_RE = re.compile(r"\|([0-9]+)(?:x([0-9]+))?(?:\s|$)")
_WIDTH_RE = re.compile(r"\bwidth=([0-9]+)(?:\b|[&;\s])", re.IGNORECASE)
_HEIGHT_RE = re.compile(r"\bheight=([0-9]+)(?:\b|[&;\s])", re.IGNORECASE)


def export_units_to_markdown_image_dimension_hint_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"])))
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
        if in_fence:
            continue
        for match in _IMAGE_RE.finditer(line):
            target = field_value(match.group(2))
            hint = _dimension_hint(target)
            if hint is None:
                continue
            width, height, style = hint
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "alt_text": field_value(match.group(1)),
                    "target": target,
                    "width": width,
                    "height": height,
                    "hint_style": style,
                    "line_number": line_number,
                    "context": field_value(line),
                }
            )
    return rows


def _dimension_hint(target: str) -> tuple[str, str, str] | None:
    size = _SIZE_RE.search(target)
    if size and (size.group(1) or size.group(2)):
        return (size.group(1) or "", size.group(2) or "", "equals")
    pipe = _PIPE_RE.search(target)
    if pipe:
        return (pipe.group(1), pipe.group(2) or "", "pipe")
    width = _WIDTH_RE.search(target)
    height = _HEIGHT_RE.search(target)
    if width or height:
        return (width.group(1) if width else "", height.group(1) if height else "", "attribute")
    return None
