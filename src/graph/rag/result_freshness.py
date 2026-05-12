"""Score RAG/search results by source or unit freshness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from math import pow
from typing import Any

_MISSING = object()
_DATE_KEYS = (
    "updated_at",
    "published_at",
    "created_at",
    "date",
    "source_date",
    "unit_date",
)


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


def _iter_result_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)

    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value

    source = _field_value(payload, "source")
    if source is not _MISSING and source is not None:
        value = _field_value(source, key)
        if value is not _MISSING:
            yield value
        metadata = _field_value(source, "metadata")
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


def _result_value(result: Any, key: str) -> Any:
    for value in _iter_result_values(result, key):
        if value is not None:
            return value
    return _MISSING


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _coerce_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime):
        raise ValueError("now must be a datetime or None")
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _validate_half_life_days(value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError("half_life_days must be a positive number")
    return float(value)


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


def _has_value(value: Any) -> bool:
    return value is not _MISSING and value is not None and (
        not isinstance(value, str) or bool(value.strip())
    )


def _freshness_score(timestamp: datetime, now: datetime, half_life_days: float) -> tuple[float, float]:
    age_seconds = max((now - timestamp).total_seconds(), 0.0)
    age_days = age_seconds / 86_400.0
    score = pow(0.5, age_days / half_life_days)
    return round(score, 6), round(age_days, 6)


def _best_timestamp(result: Any) -> tuple[tuple[datetime, str] | None, bool]:
    dated: list[tuple[datetime, str]] = []
    saw_invalid = False

    for key in _DATE_KEYS:
        for value in _iter_result_values(result, key):
            parsed = _parse_datetime(value)
            if parsed is not None:
                dated.append((parsed, key))
            elif _has_value(value):
                saw_invalid = True

    if not dated:
        return None, saw_invalid
    return max(dated, key=lambda item: item[0]), saw_invalid


def score_result_freshness(
    results: Iterable[Any],
    *,
    now: datetime | None = None,
    half_life_days: int | float = 365,
) -> list[dict[str, Any]]:
    """Return per-result freshness scores using ISO date or datetime metadata."""
    now_value = _coerce_now(now)
    half_life = _validate_half_life_days(half_life_days)

    rows = []
    for index, result in enumerate(results):
        best, saw_invalid = _best_timestamp(result)
        result_id = _result_id(result, index)
        if best is None:
            rows.append(
                {
                    "result_id": result_id,
                    "freshness_score": 0.0,
                    "age_days": None,
                    "reason": "invalid date metadata"
                    if saw_invalid
                    else "missing date metadata",
                }
            )
            continue

        timestamp, key = best
        score, age_days = _freshness_score(timestamp, now_value, half_life)
        reason = f"freshness from {key}"
        if timestamp > now_value:
            reason = f"future {key} treated as current"
        rows.append(
            {
                "result_id": result_id,
                "freshness_score": score,
                "age_days": age_days,
                "reason": reason,
            }
        )

    return rows
