"""CSV export for Markdown-embedded HTML word break tags."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "surrounding_text_preview", "before_text_preview", "after_text_preview", "id", "class"]
_WBR_RE = re.compile(r"<wbr\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)


def export_units_to_markdown_html_word_break_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["surrounding_text_preview"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _WBR_RE.finditer(content):
        values = attrs(match.group("attrs"))
        before = content[max(0, match.start() - 60) : match.start()]
        after = content[match.end() : match.end() + 60]
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "surrounding_text_preview": preview(before + after),
                "before_text_preview": preview(before),
                "after_text_preview": preview(after),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
            }
        )
    return rows
