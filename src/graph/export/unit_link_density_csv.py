"""CSV export for per-unit link density."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "content_length", "internal_wikilink_count", "markdown_link_count", "raw_url_count", "total_link_count", "links_per_1000_chars"]
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_RAW_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)


def export_units_to_link_density_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    markdown_spans = [match.span() for match in _MARKDOWN_LINK_RE.finditer(content)]
    wikilinks = len(_WIKILINK_RE.findall(content))
    markdown = len(markdown_spans)
    raw = sum(1 for match in _RAW_URL_RE.finditer(content) if not any(start <= match.start() < end for start, end in markdown_spans))
    total = wikilinks + markdown + raw
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title") or metadata(unit).get("title")),
        "content_length": len(content),
        "internal_wikilink_count": wikilinks,
        "markdown_link_count": markdown,
        "raw_url_count": raw,
        "total_link_count": total,
        "links_per_1000_chars": f"{((total / len(content)) * 1000) if content else 0:.2f}",
    }
