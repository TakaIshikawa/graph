"""CSV export for Markdown-embedded HTML microdata elements."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._markdown_html_csv import attrs, bool_attr, content_without_fences, line_number, preview, unit_context
from graph.export._report_csv import render_csv, sort_key, write_csv

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_path",
    "source",
    "line_number",
    "tag_name",
    "itemscope",
    "itemtype",
    "itemid",
    "itemprop",
    "itemref",
    "text_preview",
    "id",
    "class",
]
_MICRODATA_ATTRS = {"itemscope", "itemtype", "itemid", "itemprop", "itemref"}
_TAG_RE = re.compile(
    r"<(?P<tag>[A-Za-z][\w:.-]*)\b(?P<attrs>[^<>]*?)(?:>(?P<body>.*?)</(?P=tag)\s*>|/?>)",
    re.IGNORECASE | re.DOTALL,
)


def export_units_to_markdown_html_microdata_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag_name"]), sort_key(row["itemprop"])))
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
        values = attrs(match.group("attrs") or "")
        if not _MICRODATA_ATTRS.intersection(values):
            continue
        rows.append(
            {
                **context,
                "line_number": line_number(content, match.start()),
                "tag_name": match.group("tag").casefold(),
                "itemscope": bool_attr(values, "itemscope"),
                "itemtype": values.get("itemtype", ""),
                "itemid": values.get("itemid", ""),
                "itemprop": values.get("itemprop", ""),
                "itemref": values.get("itemref", ""),
                "text_preview": preview(match.group("body") or ""),
                "id": values.get("id", ""),
                "class": values.get("class", ""),
            }
        )
    return rows
