"""CSV export for Markdown-embedded HTML image alt metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "src", "alt", "title", "width", "height", "loading", "decoding", "missing_alt", "empty_alt", "id", "class"]
_IMG_RE = re.compile(r"<img\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_image_alt_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["src"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _IMG_RE.finditer(content):
        values = attrs(match.group("attrs"))
        rows.append({**context, "line_number": line_number(content, match.start()), "src": values.get("src", ""), "alt": values.get("alt", ""), "title": values.get("title", ""), "width": values.get("width", ""), "height": values.get("height", ""), "loading": values.get("loading", ""), "decoding": values.get("decoding", ""), "missing_alt": str("alt" not in values).lower(), "empty_alt": str("alt" in values and not values.get("alt", "")).lower(), "id": values.get("id", ""), "class": values.get("class", "")})
    return rows
