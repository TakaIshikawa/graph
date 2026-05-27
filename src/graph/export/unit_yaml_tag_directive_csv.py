"""CSV export for YAML frontmatter %TAG directives."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "handle", "prefix", "line_number"]
_TAG_RE = re.compile(r"^%TAG\s+(?P<handle>\S+)\s+(?P<prefix>\S+)\s*$")


def export_unit_yaml_tag_directive_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["handle"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    lines = str(get(unit, "content") or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            break
        match = _TAG_RE.match(line.strip())
        if match:
            rows.append({"unit_id": uid, "title": title, "handle": field_value(match.group("handle")), "prefix": field_value(match.group("prefix")), "line_number": line_number})
    return rows
