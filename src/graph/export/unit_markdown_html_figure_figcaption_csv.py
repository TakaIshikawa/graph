"""CSV export for Markdown-embedded HTML figure figcaption metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "figure_id", "figure_class", "has_figcaption", "figcaption_count", "figcaption_text_preview", "figcaption_id", "figcaption_class"]
_FIGURE_RE = re.compile(r"<figure\b(?P<attrs>[^>]*)>(?P<body>.*?)</figure\s*>|<figure\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_FIGCAPTION_RE = re.compile(r"<figcaption\b(?P<attrs>[^>]*)>(?P<body>.*?)</figcaption\s*>", re.IGNORECASE | re.DOTALL)


def export_units_to_markdown_html_figure_figcaption_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["figure_id"]), sort_key(row["figcaption_text_preview"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    content = content_without_fences(unit)
    context = unit_context(unit)
    rows: list[dict[str, str | int]] = []
    for match in _FIGURE_RE.finditer(content):
        values = attrs(match.group("attrs") or match.group("single_attrs") or "")
        captions = list(_FIGCAPTION_RE.finditer(match.group("body") or ""))
        if captions:
            for caption in captions:
                caption_values = attrs(caption.group("attrs"))
                rows.append(_row(context, content, match, values, captions, caption, caption_values))
        else:
            rows.append(_row(context, content, match, values, captions, None, {}))
    return rows


def _row(
    context: Mapping[str, str],
    content: str,
    figure_match: re.Match[str],
    values: Mapping[str, str],
    captions: list[re.Match[str]],
    caption: re.Match[str] | None,
    caption_values: Mapping[str, str],
) -> dict[str, str | int]:
    return {
        **context,
        "line_number": line_number(content, figure_match.start()),
        "figure_id": values.get("id", ""),
        "figure_class": values.get("class", ""),
        "has_figcaption": str(bool(captions)).lower(),
        "figcaption_count": len(captions),
        "figcaption_text_preview": preview(caption.group("body") if caption else ""),
        "figcaption_id": caption_values.get("id", ""),
        "figcaption_class": caption_values.get("class", ""),
    }
