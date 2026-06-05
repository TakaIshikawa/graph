"""CSV export for Markdown-embedded HTML dialog elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "id", "open", "aria_label", "aria_labelledby", "role", "text_preview", "multiline"]
_DIALOG_RE = re.compile(r"<dialog\b(?P<attrs>[^>]*)>(?P<body>.*?)</dialog\s*>|<dialog\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_dialog_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    return [
        {
            **context,
            "line_number": line_number(content, match.start()),
            "id": (values := attrs(match.group("attrs") or match.group("single_attrs") or "")).get("id", ""),
            "open": bool_attr(values, "open"),
            "aria_label": values.get("aria-label", ""),
            "aria_labelledby": values.get("aria-labelledby", ""),
            "role": values.get("role", ""),
            "text_preview": preview(match.group("body") or ""),
            "multiline": str("\n" in match.group(0)).lower(),
        }
        for match in _DIALOG_RE.finditer(content)
    ]
