"""Analyze date metadata coverage for RAG/search results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

_MISSING = object()
_UNKNOWN_SOURCE = "unknown"


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _iter_result_values(result: Any, key: str) -> Iterable[Any]:
    payload = _result_payload(result)

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


def _result_value(result: Any, key: str) -> Any:
    payload = _result_payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)

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


def _has_missing_value(value: Any) -> bool:
    return value is _MISSING or value is None or (isinstance(value, str) and not value.strip())


def _date_status(result: Any, date_keys: Sequence[str]) -> tuple[str, list[date]]:
    dates: list[date] = []
    saw_invalid = False

    for key in date_keys:
        for value in _iter_result_values(result, key):
            parsed = _parse_date(value)
            if parsed is not None:
                dates.append(parsed)
            elif not _has_missing_value(value):
                saw_invalid = True

    if dates:
        return "dated", dates
    if saw_invalid:
        return "invalid", []
    return "missing", []


def _source_label(value: Any) -> str:
    if value is _MISSING or value is None:
        return _UNKNOWN_SOURCE
    if hasattr(value, "value"):
        value = value.value
    label = " ".join(str(value).strip().split())
    return label or _UNKNOWN_SOURCE


def _ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _empty_source_counts() -> dict[str, int]:
    return {
        "total_results": 0,
        "dated_results": 0,
        "missing_date_results": 0,
        "invalid_date_results": 0,
    }


def analyze_result_date_coverage(
    results: Iterable[dict],
    *,
    date_keys: Sequence[str] = ("created_at", "updated_at", "published_at"),
) -> dict[str, Any]:
    """Return date metadata quality counts for retrieved results.

    Date fields are inspected on the result itself, result ``metadata``, nested
    ``unit`` payloads, and nested unit ``metadata``. A result is counted as
    dated when any configured date field contains an ISO date/datetime string,
    ``date``, or ``datetime`` value.
    """
    result_list = list(results)
    total = len(result_list)
    dated_results = 0
    missing_date_results = 0
    invalid_date_results = 0
    all_dates: list[date] = []
    per_source: dict[str, dict[str, int]] = defaultdict(_empty_source_counts)

    for result in result_list:
        source = _source_label(_result_value(result, "source_project"))
        source_counts = per_source[source]
        source_counts["total_results"] += 1

        status, dates = _date_status(result, date_keys)
        if status == "dated":
            dated_results += 1
            source_counts["dated_results"] += 1
            all_dates.extend(dates)
        elif status == "invalid":
            invalid_date_results += 1
            source_counts["invalid_date_results"] += 1
        else:
            missing_date_results += 1
            source_counts["missing_date_results"] += 1

    earliest_date = min(all_dates).isoformat() if all_dates else None
    latest_date = max(all_dates).isoformat() if all_dates else None

    return {
        "total_results": total,
        "dated_results": dated_results,
        "missing_date_results": missing_date_results,
        "invalid_date_results": invalid_date_results,
        "earliest_date": earliest_date,
        "latest_date": latest_date,
        "coverage_ratio": _ratio(dated_results, total),
        "per_source": {
            source: dict(counts) for source, counts in sorted(per_source.items())
        },
    }
