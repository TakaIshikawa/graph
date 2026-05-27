"""CSV export for duplicate Markdown headings within units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "heading_text", "normalized_slug", "first_line", "duplicate_line", "level"]
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def export_units_to_heading_duplicate_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["duplicate_line"]), sort_key(row["normalized_slug"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    seen: dict[str, tuple[int, int, str]] = {}
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        text = field_value(match.group(2))
        slug = _slug(text)
        if not slug:
            continue
        if slug in seen:
            first_line, _first_level, first_text = seen[slug]
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "heading_text": text or first_text,
                    "normalized_slug": slug,
                    "first_line": first_line,
                    "duplicate_line": line_number,
                    "level": level,
                }
            )
        else:
            seen[slug] = (line_number, level, text)
    return rows


def _slug(text: str) -> str:
    return re.sub(r"\s+", "-", _PUNCT_RE.sub("", text).casefold().strip()).strip("-")
