"""Summarize temporal metadata keys on relation records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_TEMPORAL_KEYS = ("created_at", "updated_at", "observed_at", "start_date", "end_date")
_ID_KEYS = ("id", "relation_id", "edge_id")


def summarize_relation_temporal_metadata(relations: Iterable[Any]) -> dict[str, Any]:
    relation_count = 0
    key_presence_counts: Counter[str] = Counter()
    valid_date_counts: Counter[str] = Counter()
    valid_datetime_counts: Counter[str] = Counter()
    invalid_temporal_values: list[dict[str, str]] = []

    for relation in relations:
        relation_count += 1
        meta = metadata(relation)
        rid = _relation_id(relation)
        for key in _TEMPORAL_KEYS:
            value = meta.get(key)
            if value in (None, ""):
                value = get(relation, key)
            if value in (None, ""):
                continue
            key_presence_counts[key] += 1
            parsed = _parse_temporal(value)
            if parsed == "date":
                valid_date_counts[key] += 1
            elif parsed == "datetime":
                valid_datetime_counts[key] += 1
            else:
                invalid_temporal_values.append({"relation_id": rid, "key": key, "value": field_value(value)})

    return {
        "relation_count": relation_count,
        "key_presence_counts": _sorted_counts(key_presence_counts),
        "valid_date_counts": _sorted_counts(valid_date_counts),
        "valid_datetime_counts": _sorted_counts(valid_datetime_counts),
        "invalid_temporal_values": sorted(invalid_temporal_values, key=lambda row: (sort_key(row["relation_id"]), sort_key(row["key"]))),
    }


def _parse_temporal(value: Any) -> str | None:
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    text = field_value(value)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return "datetime" if "T" in text or " " in text else "date"


def _relation_id(relation: Any) -> str:
    meta = metadata(relation)
    for key in _ID_KEYS:
        value = field_value(get(relation, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""


def _sorted_counts(counts: Counter[str]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts, key=sort_key)}
