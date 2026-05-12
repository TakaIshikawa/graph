"""Summarize source attribution for retrieved RAG/search results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_MISSING = object()
_UNKNOWN_SOURCE = "unknown"
_ID_KEYS = ("id", "unit_id", "source_id")
_TITLE_KEYS = ("title", "name", "book_title", "bookTitle")
_UPDATED_AT_KEYS = (
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


def _metadata(result: Any) -> dict[str, Any]:
    payload = _payload(result)
    merged: dict[str, Any] = {}

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            merged.update(unit_metadata)

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        merged.update(metadata)

    return merged


def _string(value: Any, default: str | None = None) -> str | None:
    if value is _MISSING or value is None:
        return default
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or default


def _result_id(result: Any, index: int) -> str:
    return _string(_first_value(result, _ID_KEYS), f"result-{index + 1}") or f"result-{index + 1}"


def _source_project(result: Any) -> str:
    return _string(_first_value(result, ("source_project",)), _UNKNOWN_SOURCE) or _UNKNOWN_SOURCE


def _title(result: Any) -> str | None:
    return _string(_first_value(result, _TITLE_KEYS))


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


def _latest_updated_at(result: Any) -> datetime | None:
    timestamps = [
        parsed
        for key in _UPDATED_AT_KEYS
        for value in _candidate_values(result, key)
        if (parsed := _parse_datetime(value)) is not None
    ]
    return max(timestamps) if timestamps else None


def summarize_source_attribution(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Group retrieved results by source project with compact attribution details."""
    groups: dict[str, dict[str, Any]] = {}
    metadata_counts: dict[str, Counter[str]] = defaultdict(Counter)
    latest_dates: dict[str, datetime] = {}

    for index, result in enumerate(results):
        source = _source_project(result)
        group = groups.setdefault(
            source,
            {
                "source_project": source,
                "count": 0,
                "result_ids": set(),
                "title_samples": set(),
            },
        )
        group["count"] += 1
        group["result_ids"].add(_result_id(result, index))

        title = _title(result)
        if title is not None:
            group["title_samples"].add(title)

        for key, value in _metadata(result).items():
            if value is not None and not (isinstance(value, str) and not value.strip()):
                metadata_counts[source][str(key)] += 1

        updated_at = _latest_updated_at(result)
        if updated_at is not None and (
            source not in latest_dates or updated_at > latest_dates[source]
        ):
            latest_dates[source] = updated_at

    rows = []
    for source in sorted(groups):
        group = groups[source]
        rows.append(
            {
                "source_project": source,
                "count": group["count"],
                "result_ids": sorted(group["result_ids"]),
                "title_samples": sorted(group["title_samples"])[:3],
                "metadata_key_coverage": dict(sorted(metadata_counts[source].items())),
                "latest_updated_at": (
                    latest_dates[source].isoformat() if source in latest_dates else None
                ),
            }
        )
    return rows
