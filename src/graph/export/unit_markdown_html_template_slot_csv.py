"""CSV export for Markdown-embedded HTML template and slot elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "id", "name", "shadowrootmode", "content_preview", "multiline"]
_RE = re.compile(r"<(?P<tag>template|slot)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?:template|slot)\s*>|<(?P<single_tag>template|slot)\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_template_slot_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _RE.finditer(content):
        tag = (match.group("tag") or match.group("single_tag")).casefold()
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": tag, "id": values.get("id", ""), "name": values.get("name", ""), "shadowrootmode": values.get("shadowrootmode", ""), "content_preview": preview(match.group("body") or ""), "multiline": str("\n" in match.group(0)).lower()})
    return rows
