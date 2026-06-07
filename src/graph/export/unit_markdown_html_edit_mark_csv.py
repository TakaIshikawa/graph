"""CSV export for Markdown-embedded HTML del and ins elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "cite", "datetime", "has_datetime", "text_preview", "word_count", "link_count"]
_EDIT_RE = re.compile(r"<(?P<tag>del|ins)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)\s*>", re.IGNORECASE | re.DOTALL)
_LINK_RE = re.compile(r"<a\b[^>]*\bhref\s*=", re.IGNORECASE)


def export_units_to_markdown_html_edit_mark_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["text_preview"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _EDIT_RE.finditer(content):
        body = match.group("body")
        text = preview(body)
        values = attrs(match.group("attrs"))
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "tag": match.group("tag").casefold(),
                "cite": values.get("cite", ""),
                "datetime": values.get("datetime", ""),
                "has_datetime": str("datetime" in values).lower(),
                "text_preview": text,
                "word_count": len(text.split()),
                "link_count": len(_LINK_RE.findall(body)),
            }
        )
    return rows
