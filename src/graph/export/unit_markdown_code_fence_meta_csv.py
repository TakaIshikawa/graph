"""CSV export for Markdown code fence metadata."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "fence_marker", "language", "meta", "raw_info_string"]
_FENCE_RE = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})\s*(?P<info>.*)$")


def export_unit_markdown_code_fence_meta_to_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["fence_marker"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    open_marker = ""
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        match = _FENCE_RE.match(line)
        if not match:
            continue
        marker = match.group("marker")
        if open_marker:
            if marker.startswith(open_marker[0]) and len(marker) >= len(open_marker):
                open_marker = ""
            continue
        open_marker = marker
        language, meta = _parse_info(match.group("info"))
        if meta:
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "line_number": line_number,
                    "fence_marker": marker,
                    "language": language,
                    "meta": meta,
                    "raw_info_string": field_value(match.group("info")),
                }
            )
    return rows


def _parse_info(info: str) -> tuple[str, str]:
    text = field_value(info)
    if not text:
        return "", ""
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()
    if not tokens:
        return "", ""
    language = tokens[0]
    meta = text[len(language) :].strip()
    return language, meta
