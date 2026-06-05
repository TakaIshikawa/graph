"""CSV export for Markdown-embedded HTML progress and meter elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "value", "min", "max", "low", "high", "optimum", "text_preview", "normalized_percent"]
_RE = re.compile(r"<(?P<tag>progress|meter)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?:progress|meter)\s*>|<(?P<single_tag>progress|meter)\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_progress_meter_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"])))
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
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": tag, "value": values.get("value", ""), "min": values.get("min", ""), "max": values.get("max", ""), "low": values.get("low", ""), "high": values.get("high", ""), "optimum": values.get("optimum", ""), "text_preview": preview(match.group("body") or ""), "normalized_percent": _normalized(tag, values)})
    return rows


def _num(text: str, default: float | None = None) -> float | None:
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return None


def _normalized(tag: str, values: Mapping[str, str]) -> str:
    value = _num(values.get("value", ""))
    minimum = _num(values.get("min", ""), 0.0)
    maximum = _num(values.get("max", ""), 1.0 if tag == "progress" else None)
    if value is None or minimum is None or maximum is None or maximum <= minimum:
        return ""
    percent = ((value - minimum) / (maximum - minimum)) * 100
    if percent < 0 or percent > 100:
        return ""
    return f"{percent:.2f}".rstrip("0").rstrip(".")
