"""CSV export for notebook-style cell markers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "marker_type", "language", "raw_marker"]
_PY_RE = re.compile(r"^\s*#\s*%%\s*(.*)$")
_MD_RE = re.compile(r"^\s*<!--\s*(?:#\s*)?(%%|region|endregion)(?:\b|\s)(.*?)-->\s*$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_notebook_cell_marker_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        fence = _FENCE_RE.match(line)
        if fence:
            in_fence = not in_fence
            if line.strip().casefold().startswith("```python") and "%%" in line:
                rows.append({"unit_id": uid, "title": title, "line_number": line_number, "marker_type": "fenced-cell", "language": "python", "raw_marker": field_value(line)})
            continue
        py = _PY_RE.match(line)
        md = _MD_RE.match(line) if not in_fence else None
        if py:
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "marker_type": "percent", "language": _language(py.group(1)), "raw_marker": field_value(line)})
        elif md:
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "marker_type": md.group(1).casefold(), "language": _language(md.group(2)), "raw_marker": field_value(line)})
    return rows


def _language(text: str) -> str:
    lowered = text.casefold()
    if "[markdown]" in lowered or "markdown" in lowered:
        return "markdown"
    if "python" in lowered:
        return "python"
    return ""
