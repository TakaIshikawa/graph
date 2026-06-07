"""CSV export for Markdown-embedded MathML blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "display", "alttext", "annotation_count", "identifier_count", "operator_count", "text_preview", "nested_html_present"]
_MATH_RE = re.compile(r"<math\b(?P<attrs>[^>]*)>(?P<body>.*?)</math\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<(?P<tag>[A-Za-z][\w:.-]*)\b[^>]*>", re.IGNORECASE)


def export_units_to_markdown_html_mathml_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["display"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _MATH_RE.finditer(content):
        values = attrs(match.group("attrs"))
        body = match.group("body")
        tags = [tag.group("tag").casefold() for tag in _TAG_RE.finditer(body)]
        rows.append({**context, "line_number": line_number(content, match.start()), "display": values.get("display", ""), "alttext": values.get("alttext", ""), "annotation_count": sum(tag.endswith("annotation") or tag == "annotation-xml" for tag in tags), "identifier_count": tags.count("mi"), "operator_count": tags.count("mo"), "text_preview": preview(body), "nested_html_present": str(any(tag in {"div", "span", "a", "p", "img"} for tag in tags)).lower()})
    return rows
