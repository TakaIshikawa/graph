"""Build compact freshness summaries for retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_MISSING = object()
_UNKNOWN_SOURCE = "unknown"
_BUCKETS = ("fresh", "aging", "stale", "undated")
_TIMESTAMP_KEYS = (
    "updated_at",
    "modified_at",
    "modified",
    "created_at",
    "created",
    "published_at",
    "published",
    "date",
    "timestamp",
)
_ID_KEYS = ("id", "unit_id", "source_id")
_TITLE_KEYS = ("title", "name", "book_title", "bookTitle")


def _validate_positive_days(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    return limit


def _coerce_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime):
        raise ValueError("now must be a datetime or None")
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _tuple_score(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return _MISSING


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
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING:
            yield metadata_value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING:
            yield unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            unit_metadata_value = unit_metadata.get(key, _MISSING)
            if unit_metadata_value is not _MISSING:
                yield unit_metadata_value


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        for value in _candidate_values(result, key):
            if value is not None and not (isinstance(value, str) and not value.strip()):
                return value
    return _MISSING


def _parse_datetime(value: Any) -> datetime | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(text)
            except ValueError:
                return None
            parsed = datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                tzinfo=timezone.utc,
            )
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp(result: Any) -> datetime | None:
    for key in _TIMESTAMP_KEYS:
        dates = [
            parsed
            for value in _candidate_values(result, key)
            if (parsed := _parse_datetime(value)) is not None
        ]
        if dates:
            return max(dates)
    return None


def _string_value(value: Any, default: str | None = None) -> str | None:
    if value is _MISSING or value is None:
        return default
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or default


def _source(result: Any) -> str:
    return (
        _string_value(_first_value(result, ("source_project",)), _UNKNOWN_SOURCE)
        or _UNKNOWN_SOURCE
    )


def _bucket_for(timestamp: datetime | None, now: datetime, fresh_days: int, stale_days: int) -> str:
    if timestamp is None:
        return "undated"
    age_days = max((now - timestamp).total_seconds(), 0.0) / 86_400.0
    if age_days <= fresh_days:
        return "fresh"
    if age_days <= stale_days:
        return "aging"
    return "stale"


def _score(result: Any) -> Any:
    tuple_score = _tuple_score(result)
    if tuple_score is not _MISSING:
        return tuple_score
    return _first_value(result, ("score", "final_score", "hybrid_score", "similarity"))


def _summary(result: Any, timestamp: datetime | None, bucket: str, index: int) -> dict[str, Any]:
    summary = {
        "id": _string_value(_first_value(result, _ID_KEYS), f"result-{index + 1}"),
        "title": _string_value(_first_value(result, _TITLE_KEYS)),
        "source_project": _source(result),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "bucket": bucket,
    }
    score = _score(result)
    if score is not _MISSING and score is not None:
        summary["score"] = score
    return summary


def _sort_key(item: dict[str, Any]) -> tuple[int, float, str, str, str]:
    timestamp = item["timestamp"]
    timestamp_sort = -datetime.fromisoformat(timestamp).timestamp() if timestamp else 0.0
    return (
        0 if timestamp else 1,
        timestamp_sort,
        item["source_project"],
        item["title"] or "",
        item["id"],
    )


def build_freshness_brief(
    results: Iterable[Any],
    *,
    now: datetime | None = None,
    fresh_window_days: int = 30,
    stale_window_days: int = 180,
    limit: int = 5,
) -> dict[str, Any]:
    """Summarize retrieved result freshness by bucket and source.

    Results may be dictionaries, model-like objects, or ``(unit, score)`` tuples.
    Datetimes without timezone information are treated as UTC before bucketing.
    """
    fresh_days = _validate_positive_days(fresh_window_days, "fresh_window_days")
    stale_days = _validate_positive_days(stale_window_days, "stale_window_days")
    if stale_days <= fresh_days:
        raise ValueError("stale_window_days must be greater than fresh_window_days")
    max_results = _validate_limit(limit)
    normalized_now = _coerce_now(now)

    counts: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = {bucket: Counter() for bucket in _BUCKETS}
    representatives: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in _BUCKETS}

    for index, result in enumerate(results):
        timestamp = _timestamp(result)
        bucket = _bucket_for(timestamp, normalized_now, fresh_days, stale_days)
        source = _source(result)
        counts[bucket] += 1
        sources[bucket][source] += 1
        representatives[bucket].append(_summary(result, timestamp, bucket, index))

    bucket_rows = {}
    for bucket in _BUCKETS:
        reps = sorted(representatives[bucket], key=_sort_key)[:max_results]
        bucket_rows[bucket] = {
            "count": counts[bucket],
            "source_distribution": dict(sorted(sources[bucket].items())),
            "results": reps,
        }

    return {
        "total_results": sum(counts.values()),
        "now": normalized_now.isoformat(),
        "fresh_window_days": fresh_days,
        "stale_window_days": stale_days,
        "counts": {bucket: counts[bucket] for bucket in _BUCKETS},
        "buckets": bucket_rows,
    }
