"""CSV export for Markdown-embedded HTML autocapitalize attributes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag_name", "autocapitalize", "normalized_value", "is_off", "is_none", "is_sentences", "is_words", "is_characters", "id", "class", "text_preview"]
_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9:-]*)\b([^<>]*)>", re.IGNORECASE)


def export_units_to_markdown_html_autocapitalize_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
        if "autocapitalize" not in values:
            continue
        value = values["autocapitalize"]
        normalized = value.casefold()
        rows.append({**context, "line_number": line_number(content, match.start()), "tag_name": match.group(1).casefold(), "autocapitalize": value, "normalized_value": normalized, "is_off": str(normalized == "off").lower(), "is_none": str(normalized == "none").lower(), "is_sentences": str(normalized == "sentences").lower(), "is_words": str(normalized == "words").lower(), "is_characters": str(normalized == "characters").lower(), "id": values.get("id", ""), "class": values.get("class", ""), "text_preview": _tag_preview(content, match)})
    return rows


def _tag_preview(content: str, match: re.Match[str]) -> str:
    close_match = re.search(rf"</{re.escape(match.group(1))}\s*>", content[match.end() :], re.IGNORECASE)
    return "" if not close_match else preview(content[match.end() : match.end() + close_match.start()])
