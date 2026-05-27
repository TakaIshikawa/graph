"""CSV export for directive-like HTML comments in Markdown."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "directive", "payload", "raw_comment"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_COMMENT_RE = re.compile(r"<!--(.*?)-->")
_DIRECTIVE_RE = re.compile(r"^\s*(?:(TODO|FIXME|REVIEW|graph)\s*:\s*(.*)|(@[A-Za-z][\w-]*)\s+(.*))\s*$", re.IGNORECASE)


def export_units_to_markdown_comment_directive_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["directive"])))
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
        for comment in _COMMENT_RE.finditer(line):
            body = comment.group(1).strip()
            match = _DIRECTIVE_RE.match(body)
            if match:
                directive = (match.group(1) or match.group(3) or "").casefold()
                payload = match.group(2) or match.group(4) or ""
                rows.append({"unit_id": uid, "title": title, "line_number": line_number, "directive": directive, "payload": field_value(payload), "raw_comment": field_value(comment.group(0))})
    return rows

