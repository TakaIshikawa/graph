"""CSV export for duplicate generated Markdown heading anchors."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "slug", "heading_text", "line_number", "occurrence_index", "heading_level"]
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PUNCT_RE = re.compile(r"[^\w\s-]")
_SPACE_RE = re.compile(r"[\s_]+")


def export_units_to_markdown_duplicate_heading_anchor_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["slug"]), int(row["occurrence_index"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    seen: dict[str, int] = defaultdict(int)
    all_rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        heading = field_value(match.group(2))
        slug = _slug(heading)
        seen[slug] += 1
        all_rows.append({"unit_id": uid, "title": title, "slug": slug, "heading_text": heading, "line_number": line_number, "occurrence_index": seen[slug], "heading_level": len(match.group(1))})
    duplicate_slugs = {row["slug"] for row in all_rows if seen[str(row["slug"])] > 1}
    return [row for row in all_rows if row["slug"] in duplicate_slugs]


def _slug(text: str) -> str:
    cleaned = _PUNCT_RE.sub("", text.casefold())
    return _SPACE_RE.sub("-", cleaned).strip("-")
