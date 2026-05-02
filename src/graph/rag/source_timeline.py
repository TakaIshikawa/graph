"""Summarize RAG/search result recency by source."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

_MISSING = object()
_VALID_BUCKETS = frozenset({"day", "week", "month", "year"})
_UNKNOWN_SOURCE = "unknown"


def _validate_bucket(bucket: str) -> str:
    if bucket not in _VALID_BUCKETS:
        valid = ", ".join(sorted(_VALID_BUCKETS))
        raise ValueError(f"bucket must be one of: {valid}")
    return bucket


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    return limit


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_value(result: Any, key: str) -> Any:
    value = _field_value(result, key)
    if value is not _MISSING and value is not None:
        return value

    unit = _field_value(result, "unit")
    if unit is _MISSING or unit is None:
        return value
    nested_value = _field_value(unit, key)
    if nested_value is not _MISSING:
        return nested_value
    return value


def _parse_date(value: Any) -> date | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _source_label(value: Any) -> str:
    if value is _MISSING or value is None:
        return _UNKNOWN_SOURCE
    label = " ".join(str(value).strip().split())
    return label or _UNKNOWN_SOURCE


def _bucket_start(value: date, bucket: str) -> date:
    if bucket == "day":
        return value
    if bucket == "week":
        return value.fromordinal(value.toordinal() - value.weekday())
    if bucket == "month":
        return date(value.year, value.month, 1)
    return date(value.year, 1, 1)


def _bucket_key(start: date, bucket: str) -> str:
    if bucket == "day":
        return start.isoformat()
    if bucket == "week":
        iso_year, iso_week, _ = start.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if bucket == "month":
        return f"{start.year:04d}-{start.month:02d}"
    return f"{start.year:04d}"


def build_source_timeline(
    results: Iterable[Any],
    *,
    date_key: str = "created_at",
    source_key: str = "source_project",
    bucket: str = "month",
    limit: int | None = None,
) -> dict[str, Any]:
    """Group search result counts by source and chronological time bucket.

    Results may be mappings, objects with the requested fields, or wrappers
    containing a nested ``unit`` mapping/object. Flat result fields take
    precedence over nested unit fields.
    """
    bucket_value = _validate_bucket(bucket)
    limit_value = _validate_limit(limit)

    result_list = list(results)
    counts: dict[date, Counter[str]] = defaultdict(Counter)
    skipped_missing_date = 0
    skipped_invalid_date = 0

    for result in result_list:
        raw_date = _result_value(result, date_key)
        parsed_date = _parse_date(raw_date)
        if parsed_date is None:
            if (
                raw_date is _MISSING
                or raw_date is None
                or (isinstance(raw_date, str) and not raw_date.strip())
            ):
                skipped_missing_date += 1
            else:
                skipped_invalid_date += 1
            continue

        source = _source_label(_result_value(result, source_key))
        counts[_bucket_start(parsed_date, bucket_value)][source] += 1

    all_bucket_starts = sorted(counts)
    selected_bucket_starts = (
        all_bucket_starts if limit_value is None else all_bucket_starts[:limit_value]
    )
    sources = sorted(
        {source for start in selected_bucket_starts for source in counts[start]}
    )

    buckets = []
    for start in selected_bucket_starts:
        source_counts = {source: counts[start][source] for source in sorted(counts[start])}
        buckets.append(
            {
                "bucket": _bucket_key(start, bucket_value),
                "start": start.isoformat(),
                "sources": source_counts,
                "total": sum(source_counts.values()),
            }
        )

    included_count = sum(bucket["total"] for bucket in buckets)
    candidate_count = sum(sum(counter.values()) for counter in counts.values())
    skipped_count = skipped_missing_date + skipped_invalid_date

    return {
        "buckets": buckets,
        "sources": sources,
        "stats": {
            "result_count": len(result_list),
            "included_count": included_count,
            "candidate_count": candidate_count,
            "skipped_count": skipped_count,
            "skipped_missing_date": skipped_missing_date,
            "skipped_invalid_date": skipped_invalid_date,
            "bucket": bucket_value,
            "date_key": date_key,
            "source_key": source_key,
            "limit": limit_value,
            "omitted_buckets": len(all_bucket_starts) - len(selected_bucket_starts),
        },
    }
