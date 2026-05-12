"""Audit attribution completeness for retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_MISSING = object()
_DEFAULT_FIELDS = ("source", "title", "timestamp", "url", "stable_id")
_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "source": ("source_project", "source", "source_name", "project"),
    "title": ("title", "name", "headline"),
    "author": ("author", "creator", "created_by", "owner"),
    "creator": ("creator", "author", "created_by", "owner"),
    "timestamp": (
        "updated_at",
        "published_at",
        "created_at",
        "date",
        "timestamp",
        "modified_at",
    ),
    "url": ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri"),
    "stable_id": ("id", "unit_id", "source_id", "result_id"),
}


def audit_result_attribution(
    results: Iterable[Any],
    *,
    required_fields: Iterable[str] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Return per-result and aggregate attribution coverage for retrieved results."""
    fields = _validate_required_fields(required_fields)
    row_limit = _validate_limit(limit)
    result_list = list(results)
    coverage = {field: {"present": 0, "missing": 0} for field in fields}
    source_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for index, result in enumerate(result_list):
        result_id = _result_id(result, index)
        source = _string(_first_value(result, _FIELD_KEYS["source"])) or "unknown"
        source_counts[source] += 1

        present: list[str] = []
        missing: list[str] = []
        values: dict[str, Any] = {}
        for field in fields:
            value = _field_attribution_value(result, field)
            if value is None:
                coverage[field]["missing"] += 1
                missing.append(field)
            else:
                coverage[field]["present"] += 1
                present.append(field)
                values[field] = value

        rows.append(
            {
                "result_id": result_id,
                "source": source,
                "present_fields": present,
                "missing_fields": missing,
                "values": values,
            }
        )

    return {
        "totals": {
            "result_count": len(result_list),
            "required_fields": fields,
            "complete_result_count": sum(1 for row in rows if not row["missing_fields"]),
        },
        "field_coverage": coverage,
        "source_distribution": [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "results": rows,
        "representative_rows": sorted(
            rows,
            key=lambda row: (len(row["missing_fields"]), row["source"], row["result_id"]),
        )[:row_limit],
    }


def _validate_required_fields(required_fields: Iterable[str] | None) -> list[str]:
    if required_fields is None:
        return list(_DEFAULT_FIELDS)
    fields: list[str] = []
    for field in required_fields:
        if not isinstance(field, str) or not field.strip():
            raise ValueError("required_fields must contain non-empty strings")
        normalized = field.strip().casefold()
        if normalized not in _FIELD_KEYS:
            raise ValueError(f"unsupported attribution field: {field}")
        if normalized not in fields:
            fields.append(normalized)
    if not fields:
        raise ValueError("required_fields must contain at least one field")
    return fields


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
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _field_attribution_value(result: Any, field: str) -> Any:
    value = _first_value(result, _FIELD_KEYS[field])
    if field == "timestamp":
        parsed = _parse_datetime(value)
        return parsed.isoformat() if parsed is not None else None
    return _string(value)


def _result_id(result: Any, index: int) -> str:
    return _string(_first_value(result, _FIELD_KEYS["stable_id"])) or f"result-{index + 1}"
