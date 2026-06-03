"""CSV export for inline Markdown code language hints."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "language", "code", "line_number", "start_column", "excerpt"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_SPAN_RE = re.compile(r"`([^`\n]+)`")
_HINT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_+#.-]{0,31})\s*:\s*(.+)$")


def export_units_to_markdown_inline_code_language_hint_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["start_column"]), sort_key(row["language"])))
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
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _SPAN_RE.finditer(line):
            hint = _HINT_RE.match(match.group(1).strip())
            if hint:
                rows.append({"unit_id": uid, "title": title, "language": hint.group(1).casefold(), "code": field_value(hint.group(2)), "line_number": line_number, "start_column": match.start() + 1, "excerpt": field_value(line)})
    return rows
