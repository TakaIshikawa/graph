"""Freshness scoring for retrieved evidence rows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


DATE_KEYS = ("updated_at", "published_at", "created_at", "date")


def score_evidence_freshness(results: Iterable[Any], *, now: datetime | None = None) -> list[dict[str, Any]]:
    reference = _ensure_utc(now or datetime.now(timezone.utc))
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        result_id = _result_id(result, index)
        source_project = _source_project(result)
        freshest = _freshest_date(result)
        age_days = max(0, (reference - freshest).days) if freshest else None
        bucket = _bucket(age_days)
        rows.append(
            {
                "result_id": result_id,
                "source_project": source_project,
                "freshest_date": freshest.isoformat() if freshest else None,
                "age_days": age_days,
                "freshness_bucket": bucket,
                "freshness_score": _score(age_days),
            }
        )
    rows.sort(key=lambda row: (-row["freshness_score"], row["result_id"]))
    return rows


def _freshest_date(result: Any) -> datetime | None:
    candidates: list[datetime] = []
    for container in (result, _value(result, "metadata"), _value(result, "unit"), _value(_value(result, "unit"), "metadata")):
        if container is None:
            continue
        for key in DATE_KEYS:
            parsed = _parse_datetime(_value(container, key))
            if parsed:
                candidates.append(parsed)
    return max(candidates) if candidates else None


def _result_id(result: Any, index: int) -> str:
    for key in ("result_id", "id", "source_id"):
        value = _value(result, key)
        if value not in ("", None):
            return str(value)
    unit = _value(result, "unit")
    for key in ("id", "source_id"):
        value = _value(unit, key)
        if value not in ("", None):
            return str(value)
    return str(index)


def _source_project(result: Any) -> str:
    for container in (result, _value(result, "metadata"), _value(result, "unit"), _value(_value(result, "unit"), "metadata")):
        value = _value(container, "source_project")
        if value not in ("", None):
            return str(value)
    return ""


def _value(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if value in ("", None):
        return None
    text = str(value).strip()
    for candidate in (text, f"{text}T00:00:00"):
        try:
            return _ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
        except ValueError:
            pass
    return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _bucket(age_days: int | None) -> str:
    if age_days is None:
        return "undated"
    if age_days <= 30:
        return "fresh"
    if age_days <= 90:
        return "recent"
    if age_days <= 365:
        return "aging"
    return "stale"


def _score(age_days: int | None) -> float:
    if age_days is None:
        return 0.0
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.75
    if age_days <= 365:
        return 0.4
    return 0.1
