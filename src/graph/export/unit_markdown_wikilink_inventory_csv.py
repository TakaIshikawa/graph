"""CSV export for Markdown wikilink inventory in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "target", "alias", "raw", "line"]
_WIKILINK_RE = re.compile(r"(?<!\\)\[\[([^\[\]\n]+)\]\]")


def export_units_to_markdown_wikilink_inventory_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        for match in _WIKILINK_RE.finditer(line):
            raw = match.group(0)
            body = match.group(1).strip()
            target, separator, alias = body.partition("|")
            target = field_value(target)
            if not target:
                continue
            rows.append({"unit_id": uid, "title": title, "target": target, "alias": field_value(alias) if separator else "", "raw": raw, "line": line_number})
    return rows
