"""Analyze chronological gaps in RAG/search result timelines."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_MISSING = object()
_DATE_KEYS = ("published_at", "created_at", "updated_at", "date")
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")


def analyze_result_timeline_gaps(
    results: Iterable[Any],
    *,
    gap_days: int = 90,
) -> dict[str, Any]:
    """Return date coverage and large chronological gaps for RAG results."""
    threshold = _validate_gap_days(gap_days)
    result_list = list(results)
    dated: list[tuple[date, str]] = []
    undated_count = 0

    for index, result in enumerate(result_list):
        result_id = _result_id(result, index)
        result_date = _result_date(result)
        if result_date is None:
            undated_count += 1
            continue
        dated.append((result_date, result_id))

    dated.sort(key=lambda item: (item[0], item[1]))
    year_counts = Counter(str(result_date.year) for result_date, _ in dated)
    gaps = []

    for (previous_date, previous_id), (next_date, next_id) in zip(dated, dated[1:]):
        days = (next_date - previous_date).days
        if days > threshold:
            gaps.append(
                {
                    "previous_result_id": previous_id,
                    "next_result_id": next_id,
                    "previous_date": previous_date.isoformat(),
                    "next_date": next_date.isoformat(),
                    "gap_days": days,
                }
            )

    return {
        "result_count": len(result_list),
        "dated_count": len(dated),
        "undated_count": undated_count,
        "earliest_date": dated[0][0].isoformat() if dated else None,
        "latest_date": dated[-1][0].isoformat() if dated else None,
        "year_buckets": [
            {"year": year, "count": year_counts[year]} for year in sorted(year_counts)
        ],
        "gaps": gaps,
    }


def _validate_gap_days(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("gap_days must be a non-negative integer")
    return value


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


def _parse_date(value: Any) -> date | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value
    else:
        text = _string(value)
        if text is None:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _result_date(result: Any) -> date | None:
    for key in _DATE_KEYS:
        for value in _candidate_values(result, key):
            parsed = _parse_date(value)
            if parsed is not None:
                return parsed
    return None


def _result_id(result: Any, index: int) -> str:
    return _string(_first_value(result, _ID_KEYS)) or f"result-{index + 1}"
