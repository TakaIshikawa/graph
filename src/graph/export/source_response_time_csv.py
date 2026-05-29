"""CSV export for source response timing metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, source_id, write_csv

TIMING_KEYS = ("response_time_ms", "elapsed_ms", "duration_ms", "latency_ms", "fetch_duration_ms")
_FIELDNAMES = ["source_id", "name", "timing_key", "response_time_ms", "bucket"]


def export_source_response_time_csv(sources: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    source_list = list(sources)
    rows = [row for source in source_list for row in _rows(source)]
    rows.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["timing_key"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(source: Mapping[str, Any] | object) -> list[dict[str, str]]:
    sid = source_id(source)
    name = field_value(get(source, "name") or get(source, "title") or metadata(source).get("name"))
    rows: list[dict[str, str]] = []
    for timing_key, raw in _timing_items(source):
        value = _milliseconds(raw)
        if value is None:
            continue
        rows.append({"source_id": sid, "name": name, "timing_key": timing_key, "response_time_ms": f"{value:g}", "bucket": _bucket(value)})
    return rows


def _timing_items(source: Mapping[str, Any] | object) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    meta = metadata(source)
    for key in TIMING_KEYS:
        value = get(source, key)
        if value is not None:
            items.append((key, value))
        if key in meta:
            items.append((key, meta[key]))
    return items


def _milliseconds(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    else:
        text = field_value(value).replace(",", "")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    return number if number >= 0 else None


def _bucket(milliseconds: float) -> str:
    if milliseconds < 250:
        return "fast"
    if milliseconds < 1000:
        return "moderate"
    if milliseconds < 5000:
        return "slow"
    return "very_slow"
