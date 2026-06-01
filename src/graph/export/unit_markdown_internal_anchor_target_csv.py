"""CSV inventory for targetable Markdown anchors."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "target_type", "target", "label", "duplicate_in_unit"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_CUSTOM_ID_RE = re.compile(r"\{#([A-Za-z0-9_.:-]+)\}\s*$")
_BLOCK_ID_RE = re.compile(r"(?:^|\s)\^([A-Za-z0-9_-]+)\s*$")
_NON_SLUG_RE = re.compile(r"[^a-z0-9 -]+")
_SPACE_RE = re.compile(r"\s+")


def export_units_to_markdown_internal_anchor_target_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write targetable Markdown anchors outside fenced code."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        unit_rows = _target_rows(_content(unit))
        counts = Counter(row["target"] for row in unit_rows)
        for row in unit_rows:
            row["duplicate_in_unit"] = "true" if counts[row["target"]] > 1 else "false"
            rows.append({"unit_id": unit_id(unit), "title": title, **row})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target_type"]), sort_key(row["target"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _target_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            label = field_value(_CUSTOM_ID_RE.sub("", heading.group(2)).strip().rstrip("#").strip())
            custom = _CUSTOM_ID_RE.search(heading.group(2))
            if custom:
                rows.append({"line_number": line_number, "target_type": "custom_id", "target": custom.group(1), "label": label})
            rows.append({"line_number": line_number, "target_type": "heading", "target": _slug(label), "label": label})
        block = _BLOCK_ID_RE.search(line)
        if block:
            rows.append({"line_number": line_number, "target_type": "block_id", "target": block.group(1), "label": field_value(line[: block.start()].strip())})
    return [row for row in rows if row["target"]]


def _slug(text: str) -> str:
    return _SPACE_RE.sub("-", _NON_SLUG_RE.sub("", text.casefold()).strip())
