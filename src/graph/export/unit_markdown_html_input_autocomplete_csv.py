"""CSV export for Markdown-embedded HTML autocomplete metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag", "type", "name", "autocomplete", "autocomplete_tokens", "has_section_token", "disables_autocomplete", "id", "class"]
_TAG_RE = re.compile(r"<(?P<tag>form|input|textarea|select)\b(?P<attrs>[^>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_input_autocomplete_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["name"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    inherited = ""
    for match in _TAG_RE.finditer(content):
        tag = match.group("tag").casefold()
        values = attrs(match.group("attrs"))
        if tag == "form" and "autocomplete" in values:
            inherited = values["autocomplete"]
        autocomplete = values.get("autocomplete", inherited if tag != "form" else "")
        tokens = autocomplete.split()
        rows.append({**context, "line_number": line_number(content, match.start()), "tag": tag, "type": values.get("type", ""), "name": values.get("name", ""), "autocomplete": autocomplete, "autocomplete_tokens": len(tokens), "has_section_token": str(any(token.casefold().startswith("section-") for token in tokens)).lower(), "disables_autocomplete": str(autocomplete.casefold() == "off").lower(), "id": values.get("id", ""), "class": values.get("class", "")})
    return rows
