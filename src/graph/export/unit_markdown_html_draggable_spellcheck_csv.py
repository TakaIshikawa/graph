"""CSV export for Markdown-embedded HTML draggable and spellcheck attributes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag_name", "draggable", "spellcheck", "is_draggable_true", "is_draggable_false", "is_spellcheck_true", "is_spellcheck_false", "id", "class", "text_preview"]
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)\b([^<>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_draggable_spellcheck_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    for match in _TAG_RE.finditer(content):
        values = attrs(match.group(2))
        if "draggable" not in values and "spellcheck" not in values:
            continue
        draggable = values.get("draggable", "")
        spellcheck = values.get("spellcheck", "")
        rows.append({**context, "line_number": line_number(content, match.start()), "tag_name": match.group(1).casefold(), "draggable": draggable, "spellcheck": spellcheck, "is_draggable_true": str(draggable.casefold() == "true").lower(), "is_draggable_false": str(draggable.casefold() == "false").lower(), "is_spellcheck_true": str(spellcheck.casefold() == "true").lower(), "is_spellcheck_false": str(spellcheck.casefold() == "false").lower(), "id": values.get("id", ""), "class": values.get("class", ""), "text_preview": _tag_preview(content, match)})
    return rows


def _tag_preview(content: str, match: re.Match[str]) -> str:
    close_match = re.search(rf"</{re.escape(match.group(1))}\s*>", content[match.end() :], re.IGNORECASE)
    return "" if not close_match else preview(content[match.end() : match.end() + close_match.start()])
