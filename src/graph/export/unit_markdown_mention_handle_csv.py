"""CSV export for Markdown prose @mention handles."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "raw_handle", "normalized_handle", "context", "position"]
_HANDLE_RE = re.compile(r"(?<![\w./:-])@([A-Za-z][A-Za-z0-9_.-]{1,31})(?![\w.-]*@)(?![\w.-])")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def export_units_to_markdown_mention_handle_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), int(row["position"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _HANDLE_RE.finditer(line):
            if _inside_code_span(line, match.start()) or _looks_like_url_userinfo(line, match.start()):
                continue
            raw = match.group(0).rstrip(".")
            rows.append({
                "unit_id": uid,
                "title": title,
                "source": source,
                "line_number": line_number,
                "raw_handle": raw,
                "normalized_handle": raw.casefold(),
                "context": field_value(line)[:160],
                "position": match.start() + 1,
            })
    return rows


def _inside_code_span(line: str, offset: int) -> bool:
    return line[:offset].count("`") % 2 == 1


def _looks_like_url_userinfo(line: str, offset: int) -> bool:
    prefix = line[max(0, offset - 12) : offset].casefold()
    suffix = line[offset:]
    return (prefix.endswith("http://") or prefix.endswith("https://")) and "/" in suffix
