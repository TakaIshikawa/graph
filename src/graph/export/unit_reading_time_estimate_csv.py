"""CSV export for per-unit reading time estimates."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "word_count", "estimated_minutes", "bucket", "source", "entity_type"]
_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def export_units_to_reading_time_estimate_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    words_per_minute: int = 200,
) -> str | dict[str, Any]:
    if not isinstance(words_per_minute, int) or isinstance(words_per_minute, bool) or words_per_minute <= 0:
        raise ValueError("words_per_minute must be a positive integer")

    unit_list = list(units)
    rows = sorted((_row(unit, words_per_minute) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "words_per_minute": words_per_minute,
        "bytes_written": bytes_written,
    }


def _row(unit: Mapping[str, Any] | object, words_per_minute: int) -> dict[str, str]:
    content = field_value(get(unit, "content"))
    word_count = len(_WORD_RE.findall(content))
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title") or metadata(unit).get("title")),
        "word_count": str(word_count),
        "estimated_minutes": str(math.ceil(word_count / words_per_minute) if word_count else 0),
        "bucket": _bucket(word_count, words_per_minute),
        "source": field_value(get(unit, "source_project") or metadata(unit).get("source") or metadata(unit).get("source_project")),
        "entity_type": field_value(get(unit, "source_entity_type") or get(unit, "entity_type") or metadata(unit).get("entity_type")),
    }


def _bucket(word_count: int, words_per_minute: int) -> str:
    minutes = math.ceil(word_count / words_per_minute) if word_count else 0
    if minutes == 0:
        return "empty"
    if minutes <= 5:
        return "short"
    if minutes <= 20:
        return "medium"
    return "long"
