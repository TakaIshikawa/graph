"""CSV inventory for Markdown footnote references."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "label", "line_number", "reference_count_on_line", "source_url"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEFINITION_RE = re.compile(r"^\s*\[\^([^\]\n]+)\]:")
_REFERENCE_RE = re.compile(r"\[\^([^\]\n]+)\]")


def export_units_to_markdown_footnote_reference_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write inline footnote references, excluding definitions."""
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        source_url = field_value(get(unit, "source_url") or metadata(unit).get("source_url") or metadata(unit).get("url"))
        rows.extend({"unit_id": unit_id(unit), "title": title, **row, "source_url": source_url} for row in _reference_rows(_content(unit)))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["label"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _reference_rows(content: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _DEFINITION_RE.match(line):
            continue
        matches = [match.group(1) for match in _REFERENCE_RE.finditer(line)]
        count = len(matches)
        for label in matches:
            rows.append({"label": field_value(label), "line_number": line_number, "reference_count_on_line": count})
    return rows
