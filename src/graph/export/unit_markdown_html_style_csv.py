"""CSV export for Markdown-embedded HTML style elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import field_value, render_csv, sort_key, write_csv

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_path",
    "source",
    "line_number",
    "media",
    "type",
    "nonce",
    "css_preview",
    "character_count",
    "import_rule_count",
    "url_reference_count",
    "empty_style",
    "id",
    "class",
]
_STYLE_RE = re.compile(r"<style\b(?P<attrs>[^>]*)>(?P<body>.*?)</style\s*>", re.IGNORECASE | re.DOTALL)
_IMPORT_RE = re.compile(r"@import\b", re.IGNORECASE)
_URL_RE = re.compile(r"\burl\s*\(", re.IGNORECASE)


def export_units_to_markdown_html_style_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["media"]), sort_key(row["id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _STYLE_RE.finditer(content):
        values = attrs(match.group("attrs") or "")
        css_text = field_value(match.group("body") or "")
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "media": values.get("media", ""),
                "type": values.get("type", ""),
                "nonce": values.get("nonce", ""),
                "css_preview": preview(css_text),
                "character_count": len(css_text),
                "import_rule_count": len(_IMPORT_RE.findall(css_text)),
                "url_reference_count": len(_URL_RE.findall(css_text)),
                "empty_style": str(not css_text).lower(),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
            }
        )
    return rows
