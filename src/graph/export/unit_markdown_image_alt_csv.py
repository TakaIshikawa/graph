"""CSV export for Markdown image alt text in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "image_url", "alt_text", "has_alt_text", "image_count"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)")


def export_units_to_markdown_image_alt_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["image_url"]), sort_key(row["alt_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    images: list[tuple[str, str]] = []
    for line in _content_lines(str(get(unit, "content") or data.get("content") or "")):
        images.extend((field_value(match.group(2)), field_value(match.group(1))) for match in _IMAGE_RE.finditer(line))
    image_count = len(images)
    return [
        {
            "unit_id": uid,
            "title": title,
            "image_url": image_url,
            "alt_text": alt_text,
            "has_alt_text": bool(alt_text.strip()),
            "image_count": image_count,
        }
        for image_url, alt_text in images
    ]


def _content_lines(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append(line)
    return rows
