"""Summarize source response timing metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id
from graph.export.source_response_time_csv import TIMING_KEYS


def summarize_source_response_times(sources: Iterable[Mapping[str, Any] | object], *, sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows: list[dict[str, Any]] = []
    for source in source_list:
        for timing_key, raw in _timing_items(source):
            value = _milliseconds(raw)
            if value is None:
                continue
            rows.append({"source_id": source_id(source), "timing_key": timing_key, "response_time_ms": value, "bucket": _bucket(value)})

    values = [row["response_time_ms"] for row in rows]
    rows.sort(key=lambda row: (-row["response_time_ms"], sort_key(row["source_id"]), sort_key(row["timing_key"])))
    bucket_counts = {bucket: 0 for bucket in ("fast", "moderate", "slow", "very_slow")}
    for row in rows:
        bucket_counts[row["bucket"]] += 1
    return {
        "total_sources": len(source_list),
        "sources_with_timing": len({row["source_id"] for row in rows}),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "average_ms": round(sum(values) / len(values), 2) if values else None,
        "bucket_counts": bucket_counts,
        "slow_source_samples": [
            {"source_id": row["source_id"], "timing_key": row["timing_key"], "response_time_ms": row["response_time_ms"]}
            for row in rows
            if row["bucket"] in {"slow", "very_slow"}
        ][:sample_limit],
    }


def _timing_items(source: Mapping[str, Any] | object) -> list[tuple[str, object]]:
    meta = metadata(source)
    items: list[tuple[str, object]] = []
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
