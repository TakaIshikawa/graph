"""CSV export for Obsidian-style callouts in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "callout_count", "callout_types", "folded_callout_count", "titled_callout_count", "max_callout_line_count"]
_CALLOUT_RE = re.compile(r"^\s*>\s*\[!\s*([^\]\s]+)\s*\]\s*([+-])?\s*(.*)$", re.IGNORECASE)


def export_units_to_callout_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    callouts = _callouts("" if get(unit, "content") is None else str(get(unit, "content")))
    return {
        "unit_id": unit_id(unit),
        "callout_count": len(callouts),
        "callout_types": "; ".join(sorted({kind for kind, _folded, _titled, _lines in callouts}, key=sort_key)),
        "folded_callout_count": sum(1 for _kind, folded, _titled, _lines in callouts if folded),
        "titled_callout_count": sum(1 for _kind, _folded, titled, _lines in callouts if titled),
        "max_callout_line_count": max((lines for _kind, _folded, _titled, lines in callouts), default=0),
    }


def _callouts(content: str) -> list[tuple[str, bool, bool, int]]:
    lines = content.splitlines()
    callouts: list[tuple[str, bool, bool, int]] = []
    index = 0
    while index < len(lines):
        match = _CALLOUT_RE.match(lines[index])
        if not match:
            index += 1
            continue
        kind = field_value(match.group(1)).casefold()
        folded = bool(match.group(2))
        titled = bool(field_value(match.group(3)))
        line_count = 1
        index += 1
        while index < len(lines) and lines[index].lstrip().startswith(">") and not _CALLOUT_RE.match(lines[index]):
            line_count += 1
            index += 1
        callouts.append((kind, folded, titled, line_count))
    return callouts
