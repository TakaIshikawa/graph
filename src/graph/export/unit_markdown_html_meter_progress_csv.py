"""CSV export for Markdown-embedded HTML meter and progress elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag_name", "value", "min", "max", "low", "high", "optimum", "has_value", "id", "class", "text_preview"]
_RE = re.compile(r"<(?P<tag>meter|progress)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?:meter|progress)\s*>|<(?P<single_tag>meter|progress)\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_meter_progress_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag_name"]), sort_key(row["id"])))
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
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        rows.append({**context, "line_number": line_number(content, match.start()), "tag_name": (match.group("tag") or match.group("single_tag")).casefold(), "value": values.get("value", ""), "min": values.get("min", ""), "max": values.get("max", ""), "low": values.get("low", ""), "high": values.get("high", ""), "optimum": values.get("optimum", ""), "has_value": str("value" in values).lower(), "id": values.get("id", ""), "class": values.get("class", ""), "text_preview": preview(match.group("body") or "")})
    return rows
