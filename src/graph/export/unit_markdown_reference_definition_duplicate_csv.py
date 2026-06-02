"""CSV export for duplicate Markdown reference definitions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "label", "normalized_label", "first_line_number", "duplicate_line_number", "duplicate_target"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^[ \t]{0,3}\[([^\]\n]+)]\s*:\s*(\S.*)?$")


def export_unit_markdown_reference_definition_duplicates_to_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["normalized_label"]), int(row["duplicate_line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    first_by_label: dict[str, tuple[str, int]] = {}
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or metadata(unit).get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _DEF_RE.match(line)
        if not match:
            continue
        label = field_value(match.group(1))
        normalized = _normalize_label(label)
        target = field_value(match.group(2) or "")
        if normalized in first_by_label:
            first_label, first_line = first_by_label[normalized]
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "label": first_label,
                    "normalized_label": normalized,
                    "first_line_number": first_line,
                    "duplicate_line_number": line_number,
                    "duplicate_target": target,
                }
            )
        else:
            first_by_label[normalized] = (label, line_number)
    return rows


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", field_value(value)).casefold()
