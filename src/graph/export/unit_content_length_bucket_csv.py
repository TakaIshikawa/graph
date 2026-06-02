"""CSV export for per-unit content length buckets."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "char_count", "word_count", "length_bucket"]
_WORD_RE = re.compile(r"\S+")


def export_units_to_content_length_bucket_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [_row(unit) for unit in unit_list]
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, Any]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    char_count = len(content)
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title")),
        "char_count": char_count,
        "word_count": len(_WORD_RE.findall(content)),
        "length_bucket": _bucket(char_count),
    }


def _bucket(length: int) -> str:
    if length == 0:
        return "empty"
    if length <= 280:
        return "short"
    if length <= 2_000:
        return "medium"
    if length <= 10_000:
        return "long"
    return "very_long"
