"""CSV export for Markdown-embedded HTML charset declarations."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "charset", "http_equiv", "content", "declares_charset", "id", "class"]
_META_RE = re.compile(r"<meta\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_meta_charset_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["charset"]), sort_key(row["http_equiv"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _META_RE.finditer(content):
        values = attrs(match.group("attrs"))
        charset = values.get("charset", "")
        http_equiv = values.get("http-equiv", "")
        meta_content = values.get("content", "")
        declares_charset = bool(charset) or (http_equiv.casefold() == "content-type" and "charset=" in meta_content.casefold())
        if not declares_charset and http_equiv.casefold() != "content-type":
            continue
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "charset": charset,
                "http_equiv": http_equiv,
                "content": meta_content,
                "declares_charset": str(declares_charset).lower(),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
            }
        )
    return rows
