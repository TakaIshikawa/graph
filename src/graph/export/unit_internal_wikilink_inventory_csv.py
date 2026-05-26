"""CSV export for Obsidian-style internal wikilinks in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "wikilink_count", "unique_targets", "labeled_link_count", "empty_target_count"]
_WIKILINK_RE = re.compile(r"\[\[([^\]]*)\]\]")
_MARKDOWN_METADATA_KEYS = {"markdown", "note", "notes", "description", "summary", "body", "content"}


def export_units_to_internal_wikilink_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    links = [link for text in _markdown_texts(unit) for link in _links(text)]
    targets = sorted({target for target, _label in links if target}, key=sort_key)
    return {
        "unit_id": unit_id(unit),
        "wikilink_count": len(links),
        "unique_targets": "; ".join(targets),
        "labeled_link_count": sum(1 for _target, label in links if label),
        "empty_target_count": sum(1 for target, _label in links if not target),
    }


def _markdown_texts(unit: Mapping[str, Any] | object) -> list[str]:
    texts = [field_value(get(unit, "content"))]
    for key, value in metadata(unit).items():
        if field_value(key).casefold() in _MARKDOWN_METADATA_KEYS:
            texts.extend(field_value(item) for item in flatten_values(value))
    return [text for text in texts if text]


def _links(text: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for match in _WIKILINK_RE.finditer(text):
        raw = match.group(1).strip()
        target, separator, label = raw.partition("|")
        links.append((field_value(target), field_value(label) if separator else ""))
    return links
