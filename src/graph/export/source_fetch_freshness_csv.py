"""CSV export for source fetch freshness buckets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.export._report_csv import get, metadata, parse_datetime, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["age_bucket", "source_count", "oldest_timestamp", "newest_timestamp", "source_ids"]
_TIMESTAMP_KEYS = ("fetched_at", "last_fetched_at", "imported_at", "ingested_at", "updated_at", "created_at")
_UNKNOWN = "unknown"


def export_source_fetch_freshness_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> str | dict[str, Any]:
    """Return or write source counts grouped by best available fetch/import timestamp."""
    source_list = list(sources)
    rows = _freshness_rows(source_list, now=now or datetime.now(timezone.utc))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _freshness_rows(sources: list[Mapping[str, Any] | object], *, now: datetime) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"timestamps": [], "ids": set()})
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    for source in sources:
        timestamp = _source_timestamp(source)
        bucket = _age_bucket(timestamp, now)
        if timestamp is not None:
            buckets[bucket]["timestamps"].append(timestamp)
        if source_id(source):
            buckets[bucket]["ids"].add(source_id(source))

    rows: list[dict[str, str | int]] = []
    for bucket in sorted(buckets, key=_bucket_sort_key):
        timestamps = sorted(buckets[bucket]["timestamps"])
        rows.append(
            {
                "age_bucket": bucket,
                "source_count": len(buckets[bucket]["ids"]) if buckets[bucket]["ids"] else len(timestamps),
                "oldest_timestamp": timestamps[0].isoformat() if timestamps else "",
                "newest_timestamp": timestamps[-1].isoformat() if timestamps else "",
                "source_ids": "; ".join(sorted(buckets[bucket]["ids"], key=sort_key)),
            }
        )
    return rows


def _source_timestamp(source: Mapping[str, Any] | object) -> datetime | None:
    for key in _TIMESTAMP_KEYS:
        parsed = parse_datetime(get(source, key))
        if parsed is None:
            parsed = parse_datetime(metadata(source).get(key))
        if parsed is not None:
            return parsed
    return None


def _age_bucket(timestamp: datetime | None, now: datetime) -> str:
    if timestamp is None:
        return _UNKNOWN
    age_days = max(0, (now - timestamp).days)
    if age_days <= 7:
        return "0-7_days"
    if age_days <= 30:
        return "8-30_days"
    if age_days <= 90:
        return "31-90_days"
    return "91+_days"


def _bucket_sort_key(bucket: str) -> tuple[int, tuple[str, str]]:
    order = {"0-7_days": 0, "8-30_days": 1, "31-90_days": 2, "91+_days": 3, _UNKNOWN: 4}
    return (order.get(bucket, 99), sort_key(bucket))

