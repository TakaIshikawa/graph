"""CSV export for Markdown-embedded HTML anchor rel security metadata."""

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
    "href",
    "target",
    "rel",
    "has_noopener",
    "has_noreferrer",
    "opens_new_context",
    "unsafe_blank_target",
    "text_preview",
    "id",
    "class",
]
_A_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a\s*>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_anchor_rel_security_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["href"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _A_RE.finditer(content):
        values = attrs(match.group("attrs"))
        rel_tokens = {token.casefold() for token in values.get("rel", "").split()}
        opens_new_context = values.get("target", "").casefold() == "_blank"
        has_noopener = "noopener" in rel_tokens
        has_noreferrer = "noreferrer" in rel_tokens
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "href": values.get("href", ""),
                "target": values.get("target", ""),
                "rel": values.get("rel", ""),
                "has_noopener": str(has_noopener).lower(),
                "has_noreferrer": str(has_noreferrer).lower(),
                "opens_new_context": str(opens_new_context).lower(),
                "unsafe_blank_target": str(opens_new_context and not has_noopener).lower(),
                "text_preview": preview(match.group("body") or ""),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
            }
        )
    return rows
