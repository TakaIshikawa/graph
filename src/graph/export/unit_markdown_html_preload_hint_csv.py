"""CSV export for Markdown-embedded HTML resource hint metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "href", "rel", "as_attr", "type", "crossorigin", "media", "hint_kind", "id", "class"]
_LINK_RE = re.compile(r"<link\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_HINTS = ("preload", "modulepreload", "preconnect", "dns-prefetch", "prefetch", "prerender")


def export_units_to_markdown_html_preload_hint_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["hint_kind"]), sort_key(row["href"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _LINK_RE.finditer(content):
        values = attrs(match.group("attrs"))
        rel_tokens = {token.casefold() for token in values.get("rel", "").split()}
        hint_kind = next((hint for hint in _HINTS if hint in rel_tokens), "")
        if not hint_kind:
            continue
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "href": values.get("href", ""),
                "rel": values.get("rel", ""),
                "as_attr": values.get("as", ""),
                "type": values.get("type", ""),
                "crossorigin": values.get("crossorigin", ""),
                "media": values.get("media", ""),
                "hint_kind": hint_kind,
                "id": values.get("id", ""),
                "class": values.get("class", ""),
            }
        )
    return rows
