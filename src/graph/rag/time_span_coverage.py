"""Analyze timestamp coverage across retrieved RAG results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_MISSING = object()
_TIMESTAMP_KEYS = (
    "updated_at",
    "published_at",
    "created_at",
    "date",
    "timestamp",
    "modified_at",
    "last_seen_at",
)
_ID_KEYS = ("id", "unit_id", "source_id", "result_id")
_SOURCE_KEYS = ("source_project", "source", "source_name", "project")


def analyze_time_span_coverage(
    results: Iterable[Any],
    *,
    now: datetime | None = None,
    bucket: str = "year",
    limit: int = 10,
) -> dict[str, Any]:
    """Return deterministic date bucket, source, and representative coverage details."""
    if bucket not in {"year", "month"}:
        raise ValueError("bucket must be 'year' or 'month'")
    row_limit = _validate_limit(limit)
    reference = _ensure_utc(now or datetime.now(timezone.utc))
    result_list = list(results)

    bucket_counts: Counter[str] = Counter()
    bucket_sources: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()
    representatives: list[dict[str, Any]] = []
    dated: list[datetime] = []
    undated_count = 0

    for index, result in enumerate(result_list):
        result_id = _result_id(result, index)
        source = _source(result)
        source_counts[source] += 1
        timestamp = _latest_timestamp(result)
        if timestamp is None:
            undated_count += 1
            representatives.append(
                {
                    "result_id": result_id,
                    "source": source,
                    "timestamp": None,
                    "bucket": None,
                    "age_days": None,
                }
            )
            continue

        dated.append(timestamp)
        bucket_key = _bucket_key(timestamp, bucket)
        bucket_counts[bucket_key] += 1
        bucket_sources[bucket_key][source] += 1
        representatives.append(
            {
                "result_id": result_id,
                "source": source,
                "timestamp": timestamp.isoformat(),
                "bucket": bucket_key,
                "age_days": max(0, (reference - timestamp).days),
            }
        )

    representatives.sort(
        key=lambda row: (
            row["timestamp"] is None,
            row["timestamp"] or "",
            row["source"],
            row["result_id"],
        )
    )

    return {
        "totals": {
            "result_count": len(result_list),
            "dated_count": len(dated),
            "undated_count": undated_count,
            "bucket": bucket,
        },
        "earliest_timestamp": min(dated).isoformat() if dated else None,
        "latest_timestamp": max(dated).isoformat() if dated else None,
        "bucket_counts": [
            {"bucket": key, "count": bucket_counts[key]}
            for key in sorted(bucket_counts)
        ],
        "source_distribution": [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "bucket_source_distribution": {
            key: [
                {"source": source, "count": count}
                for source, count in sorted(bucket_sources[key].items(), key=lambda item: (-item[1], item[0]))
            ]
            for key in sorted(bucket_sources)
        },
        "representative_rows": representatives[:row_limit],
    }


def _validate_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    return limit


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        metadata = _field_value(unit, "metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        for value in _candidate_values(result, key):
            if _string(value) is not None:
                return value
    return _MISSING


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _parse_datetime(value: Any) -> datetime | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        text = _string(value)
        if text is None:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(text)
            except ValueError:
                return None
            parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    return _ensure_utc(parsed)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _latest_timestamp(result: Any) -> datetime | None:
    timestamps = [
        parsed
        for key in _TIMESTAMP_KEYS
        for value in _candidate_values(result, key)
        if (parsed := _parse_datetime(value)) is not None
    ]
    return max(timestamps) if timestamps else None


def _result_id(result: Any, index: int) -> str:
    return _string(_first_value(result, _ID_KEYS)) or f"result-{index + 1}"


def _source(result: Any) -> str:
    return _string(_first_value(result, _SOURCE_KEYS)) or "unknown"


def _bucket_key(timestamp: datetime, bucket: str) -> str:
    if bucket == "year":
        return f"{timestamp.year:04d}"
    return f"{timestamp.year:04d}-{timestamp.month:02d}"
