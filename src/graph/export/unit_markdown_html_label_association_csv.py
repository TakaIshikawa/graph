"""CSV export for Markdown-embedded HTML label association metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "for_attr", "has_for", "wraps_control", "text_preview", "id", "class"]
_LABEL_RE = re.compile(r"<label\b(?P<attrs>[^>]*)>(?P<body>.*?)</label\s*>", re.IGNORECASE | re.DOTALL)
_CONTROL_RE = re.compile(r"<\s*(input|select|textarea|button)\b", re.IGNORECASE)


def export_units_to_markdown_html_label_association_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["for_attr"]), sort_key(row["text_preview"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _LABEL_RE.finditer(content):
        values = attrs(match.group("attrs"))
        body = match.group("body") or ""
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "for_attr": values.get("for", ""),
                "has_for": str("for" in values).lower(),
                "wraps_control": str(bool(_CONTROL_RE.search(body))).lower(),
                "text_preview": preview(body),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
            }
        )
    return rows
