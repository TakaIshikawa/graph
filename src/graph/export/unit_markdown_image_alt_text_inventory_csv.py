"""CSV inventory for Markdown image alt text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "alt_text", "destination", "title_text", "is_empty_alt"]
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+(\"[^\"]*\"|'[^']*'))?\)")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def export_units_to_markdown_image_alt_text_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per inline Markdown image outside fenced code."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        rows.extend(
            {"unit_id": unit_id(unit), "title": title, **row}
            for row in _image_rows(str(get(unit, "content") or metadata(unit).get("content") or ""))
        )
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["destination"]), sort_key(row["alt_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _image_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _IMAGE_RE.finditer(line):
            title = match.group(3) or ""
            rows.append(
                {
                    "line_number": line_number,
                    "alt_text": field_value(match.group(1)),
                    "destination": field_value(match.group(2)),
                    "title_text": field_value(title[1:-1] if title else ""),
                    "is_empty_alt": "true" if not field_value(match.group(1)) else "false",
                }
            )
    return rows
