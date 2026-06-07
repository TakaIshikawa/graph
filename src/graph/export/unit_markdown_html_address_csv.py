"""CSV export for Markdown-embedded HTML address elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_path",
    "source",
    "line_number",
    "id",
    "class",
    "text_preview",
    "link_count",
    "email_count",
    "tel_count",
    "has_nested_html",
]
_ADDRESS_RE = re.compile(r"<address\b(?P<attrs>[^>]*)>(?P<body>.*?)</address\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[A-Za-z][^>]*>")
_HREF_RE = re.compile(r"""<a\b[^>]*\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""", re.IGNORECASE)


def export_units_to_markdown_html_address_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    rows: list[dict[str, str | int]] = []
    for match in _ADDRESS_RE.finditer(content):
        body = match.group("body")
        values = attrs(match.group("attrs"))
        hrefs = [_href(match) for match in _HREF_RE.finditer(body)]
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
                "text_preview": preview(body),
                "link_count": len(hrefs),
                "email_count": sum(href.casefold().startswith("mailto:") for href in hrefs),
                "tel_count": sum(href.casefold().startswith("tel:") for href in hrefs),
                "has_nested_html": str(bool(_TAG_RE.search(body))).lower(),
            }
        )
    return rows


def _href(match: re.Match[str]) -> str:
    return next((value for value in match.groups() if value is not None), "")
