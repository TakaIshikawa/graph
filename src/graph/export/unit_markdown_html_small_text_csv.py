"""CSV export for Markdown-embedded HTML small elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "id", "class", "text_preview", "word_count", "link_count", "contains_copyright", "contains_license"]
_SMALL_RE = re.compile(r"<small\b(?P<attrs>[^>]*)>(?P<body>.*?)</small\s*>", re.IGNORECASE | re.DOTALL)
_LINK_RE = re.compile(r"<a\b[^>]*\bhref\s*=", re.IGNORECASE)


def export_units_to_markdown_html_small_text_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["id"]), sort_key(row["text_preview"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _SMALL_RE.finditer(content):
        body = match.group("body")
        text = preview(body)
        lowered = text.casefold()
        values = attrs(match.group("attrs"))
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
                "text_preview": text,
                "word_count": len(text.split()),
                "link_count": len(_LINK_RE.findall(body)),
                "contains_copyright": str("copyright" in lowered or "©" in text).lower(),
                "contains_license": str("license" in lowered or "licensed" in lowered).lower(),
            }
        )
    return rows
