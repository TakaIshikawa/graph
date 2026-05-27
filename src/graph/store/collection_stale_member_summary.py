"""Summarize stale member units inside collections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_COLLECTION_ID_KEYS = ("id", "collection_id", "source_id")
_MEMBER_KEYS = ("members", "member_ids", "items", "unit_ids", "units")
_TIMESTAMP_KEYS = ("updated_at", "modified_at", "created_at")


def summarize_collection_stale_members(
    collections: Iterable[Any],
    *,
    cutoff_date: date | datetime | str | None = None,
    max_age_days: int | None = None,
    reference_date: date | datetime | str | None = None,
    sample_limit: int = 5,
) -> dict[str, Any]:
    cutoff = _cutoff(cutoff_date, max_age_days, reference_date)
    rows = list(collections)
    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, Any]] = []
    stale_collection_count = stale_member_count = invalid_timestamp_count = 0

    for index, collection in enumerate(rows):
        collection_id = _collection_id(collection, index)
        stale_for_collection = 0
        for member in _members(collection):
            timestamp_value = _timestamp_value(member)
            timestamp = _parse_date(timestamp_value)
            member_id = _member_id(member)
            if timestamp is None:
                invalid_timestamp_count += 1
                if len(invalid_samples) < max(0, sample_limit):
                    invalid_samples.append({"collection_id": collection_id, "member_id": member_id, "timestamp": field_value(timestamp_value)})
                continue
            if timestamp < cutoff:
                stale_for_collection += 1
                stale_member_count += 1
                if len(samples) < max(0, sample_limit):
                    samples.append({"collection_id": collection_id, "member_id": member_id, "timestamp": timestamp.isoformat()})
        if stale_for_collection:
            stale_collection_count += 1
            counts[collection_id] = stale_for_collection

    return {
        "collection_count": len(rows),
        "stale_collection_count": stale_collection_count,
        "stale_member_count": stale_member_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "counts_by_collection": [{"collection_id": key, "stale_member_count": counts[key]} for key in sorted(counts, key=sort_key)],
        "samples": sorted(samples, key=lambda row: (sort_key(row["collection_id"]), sort_key(row["member_id"])))[: max(0, sample_limit)],
        "invalid_timestamp_samples": sorted(invalid_samples, key=lambda row: (sort_key(row["collection_id"]), sort_key(row["member_id"])))[: max(0, sample_limit)],
    }


def _cutoff(cutoff_date: date | datetime | str | None, max_age_days: int | None, reference_date: date | datetime | str | None) -> date:
    if cutoff_date is not None and max_age_days is not None:
        raise ValueError("provide cutoff_date or max_age_days, not both")
    if cutoff_date is None and max_age_days is None:
        raise ValueError("cutoff_date or max_age_days is required")
    if max_age_days is not None:
        if max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        ref = _parse_date(reference_date) or date.today()
        return ref - timedelta(days=max_age_days)
    parsed = _parse_date(cutoff_date)
    if parsed is None:
        raise ValueError("cutoff_date must be a valid date")
    return parsed


def _members(collection: Any) -> list[Any]:
    meta = metadata(collection)
    values: list[Any] = []
    for key in _MEMBER_KEYS:
        values.extend(_as_list(get(collection, key)))
        values.extend(_as_list(meta.get(key)))
    return values


def _timestamp_value(member: Any) -> Any:
    meta = metadata(member)
    for key in _TIMESTAMP_KEYS:
        value = get(member, key)
        if value is not None:
            return value
        if key in meta:
            return meta.get(key)
    return None


def _collection_id(collection: Any, index: int) -> str:
    meta = metadata(collection)
    for key in _COLLECTION_ID_KEYS:
        value = field_value(get(collection, key)) or field_value(meta.get(key))
        if value:
            return value
    return str(index)


def _member_id(member: Any) -> str:
    meta = metadata(member)
    for key in ("id", "unit_id", "member_id", "source_id"):
        value = field_value(get(member, key)) or field_value(meta.get(key))
        if value:
            return value
    return field_value(member)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [] if value is None else [value]


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = field_value(value)
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
