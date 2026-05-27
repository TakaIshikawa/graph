"""CSV export for common Markdown admonition syntaxes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "admonition_type", "syntax", "line_number", "preview"]
_PATTERNS = [
    ("obsidian", re.compile(r"^\s*>\s*\[!\s*([^\]\s]+)\s*\]\s*(.*)$", re.IGNORECASE)),
    ("mkdocs", re.compile(r'^\s*!!!\s+([A-Za-z0-9_-]+)\s*("([^"]*)")?\s*$', re.IGNORECASE)),
    ("container", re.compile(r"^\s*:::\s*([A-Za-z0-9_-]+)\s*(.*)$", re.IGNORECASE)),
]


def export_units_to_markdown_admonition_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            for syntax, pattern in _PATTERNS:
                match = pattern.match(line)
                if match:
                    preview = field_value(match.group(3) if syntax == "mkdocs" else match.group(2))
                    rows.append({"unit_id": unit_id(unit), "title": title, "admonition_type": field_value(match.group(1)).casefold(), "syntax": syntax, "line_number": line_number, "preview": preview[:80]})
                    break
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}
