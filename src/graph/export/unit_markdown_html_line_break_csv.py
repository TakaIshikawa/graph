"""CSV export for Markdown-embedded HTML hr, br, and wbr elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "id", "class", "role", "aria_hidden", "surrounding_text_preview"]
_BREAK_RE = re.compile(r"<(?P<tag>hr|br|wbr)\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)


def export_units_to_markdown_html_line_break_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    for match in _BREAK_RE.finditer(content):
        values = attrs(match.group("attrs"))
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "tag": match.group("tag").casefold(),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
                "role": values.get("role", ""),
                "aria_hidden": values.get("aria-hidden", ""),
                "surrounding_text_preview": preview(content[max(0, match.start() - 80) : match.end() + 80]),
            }
        )
    return rows
