"""CSV export for Markdown emoji shortcodes."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "shortcode", "line_number", "context"]
_SHORTCODE_RE = re.compile(r"(?<![\w:]):([a-z0-9_+-]+):(?![\w:])")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def export_units_to_markdown_emoji_shortcodes_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["shortcode"])))
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
        for match in _SHORTCODE_RE.finditer(line):
            if _inside_code_span(line, match.start()):
                continue
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "shortcode": f":{match.group(1)}:",
                    "line_number": line_number,
                    "context": field_value(line)[:160],
                }
            )
    return rows


def _inside_code_span(line: str, offset: int) -> bool:
    return line[:offset].count("`") % 2 == 1
