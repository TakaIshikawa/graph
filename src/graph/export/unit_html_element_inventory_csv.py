"""CSV export for embedded HTML elements in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "html_element_count", "distinct_tags", "link_tag_count", "image_tag_count", "unsafe_tag_count"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"</?\s*([a-zA-Z][a-zA-Z0-9:-]*)\b[^>]*>")
_UNSAFE = {"script", "style", "iframe", "object", "embed", "form"}


def export_units_to_html_element_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, int | str]:
    tags = [match.group(1).casefold() for match in _TAG_RE.finditer(_strip_fenced("" if get(unit, "content") is None else str(get(unit, "content"))))]
    return {
        "unit_id": unit_id(unit),
        "html_element_count": len(tags),
        "distinct_tags": "; ".join(sorted(set(tags), key=sort_key)),
        "link_tag_count": sum(1 for tag in tags if tag == "a"),
        "image_tag_count": sum(1 for tag in tags if tag == "img"),
        "unsafe_tag_count": sum(1 for tag in tags if tag in _UNSAFE),
    }


def _strip_fenced(content: str) -> str:
    kept: list[str] = []
    fence = ""
    for line in content.splitlines():
        match = _FENCE_RE.match(line)
        marker = match.group(1) if match else ""
        if marker and not fence:
            fence = marker[0]
            continue
        if fence and line.lstrip().startswith(fence * 3):
            fence = ""
            continue
        if not fence:
            kept.append(line)
    return "\n".join(kept)
