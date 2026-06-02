"""Summarize source fetch durations into buckets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_DURATION_KEYS = ("fetch_duration_ms", "response_time_ms", "elapsed_ms", "duration_ms")
_TIMING_KEYS = ("timing", "timings", "fetch_timing", "metrics")
_BUCKETS = ("<100ms", "100-499ms", "500-999ms", "1-4.999s", ">=5s")


def summarize_source_fetch_duration_buckets(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    valid = [row for row in rows if row["duration_ms"] is not None]
    bucket_counts = Counter({bucket: 0 for bucket in _BUCKETS})
    bucket_counts.update(_bucket(row["duration_ms"]) for row in valid)
    slow_rows = [row for row in valid if row["duration_ms"] >= 5000]
    limit = max(0, sample_limit)
    slow_samples = [
        {"source_id": row["source_id"], "duration_ms": row["duration_ms"], "bucket": row["bucket"]}
        for row in sorted(slow_rows, key=lambda row: sort_key(row["source_id"]))[:limit]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_fetch_duration": len(valid),
        "invalid_duration_count": sum(1 for row in rows if row["invalid"]),
        "bucket_counts": dict(bucket_counts),
        "slow_source_count": len(slow_rows),
        "slow_samples": slow_samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    raw = _duration_value(source)
    duration = _parse_duration_ms(raw)
    invalid = raw is not None and duration is None
    return {
        "source_id": source_id(source) or str(index),
        "duration_ms": duration,
        "invalid": invalid,
        "bucket": _bucket(duration) if duration is not None else "",
    }


def _duration_value(source: Mapping[str, Any] | object) -> object:
    data = metadata(source)
    for container in (source, data):
        for key in _DURATION_KEYS:
            value = get(container, key) if container is source else container.get(key)
            if field_value(value):
                return value
        for timing_key in _TIMING_KEYS:
            timing = get(container, timing_key) if container is source else container.get(timing_key)
            if isinstance(timing, Mapping):
                for key in _DURATION_KEYS:
                    value = timing.get(key)
                    if field_value(value):
                        return value
    return None


def _parse_duration_ms(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if value >= 0 else None
    text = field_value(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _bucket(duration_ms: float) -> str:
    if duration_ms < 100:
        return "<100ms"
    if duration_ms < 500:
        return "100-499ms"
    if duration_ms < 1000:
        return "500-999ms"
    if duration_ms < 5000:
        return "1-4.999s"
    return ">=5s"
