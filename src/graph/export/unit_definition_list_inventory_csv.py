"""CSV export for Markdown definition lists in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "definition_term_count", "definition_line_count", "multi_definition_term_count", "max_definition_lines_per_term"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^\s*:\s+")


def export_units_to_definition_list_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, int | str]:
    groups = _groups("" if get(unit, "content") is None else str(get(unit, "content")))
    return {
        "unit_id": unit_id(unit),
        "definition_term_count": len(groups),
        "definition_line_count": sum(groups),
        "multi_definition_term_count": sum(1 for count in groups if count > 1),
        "max_definition_lines_per_term": max(groups, default=0),
    }


def _groups(content: str) -> list[int]:
    lines = _strip_fenced(content).splitlines()
    groups: list[int] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip() or _DEF_RE.match(lines[index]):
            index += 1
            continue
        count = 0
        probe = index + 1
        while probe < len(lines) and _DEF_RE.match(lines[probe]):
            count += 1
            probe += 1
        if count:
            groups.append(count)
            index = probe
        else:
            index += 1
    return groups


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
