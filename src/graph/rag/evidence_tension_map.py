"""Map metadata value tensions across retrieved RAG results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_DEFAULT_FIELDS = ("status", "rating", "sentiment", "answer", "stance", "outcome")
_ID_KEYS = ("id", "unit_id", "source_id")


def _validate_non_negative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _field_names(fields: Iterable[str] | None) -> tuple[str, ...]:
    if fields is None:
        return _DEFAULT_FIELDS
    names = tuple(fields)
    for field in names:
        if not isinstance(field, str) or not field.strip():
            raise ValueError("fields must contain non-empty field names")
    return tuple(field.strip() for field in names)


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


def _result_value(result: Any, key: str) -> Any:
    payload = _payload(result)
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


def _dotted_value(result: Any, field: str) -> Any:
    if "." not in field:
        return _result_value(result, field)

    current: Any = _payload(result)
    for part in field.split("."):
        if part == "metadata":
            current = _field_value(current, "metadata")
        else:
            current = _field_value(current, part)
            if current is _MISSING:
                metadata = _field_value(_payload(result), "metadata")
                if isinstance(metadata, Mapping):
                    current = metadata.get(part, _MISSING)
        if current is _MISSING or current is None:
            return _MISSING
    return current


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _normalize_value(value: Any) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    return text.casefold()


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source_project(result: Any) -> str:
    return _string_value(_result_value(result, "source_project")) or "unknown"


def map_evidence_tensions(
    results: Iterable[Any],
    *,
    fields: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return fields whose normalized metadata values disagree across results."""
    field_names = _field_names(fields)
    limit_value = _validate_non_negative_int(limit, "limit")
    result_list = list(results)
    buckets: dict[str, dict[str, Any]] = {
        field: {"counts": Counter(), "representatives": defaultdict(list), "source_projects": set()}
        for field in field_names
    }

    for index, result in enumerate(result_list):
        result_id = _result_id(result, index)
        source_project = _source_project(result)
        for field in field_names:
            normalized = _normalize_value(_dotted_value(result, field))
            if normalized is None:
                continue
            bucket = buckets[field]
            bucket["counts"][normalized] += 1
            bucket["source_projects"].add(source_project)
            bucket["representatives"][normalized].append(result_id)

    rows = []
    total = max(len(result_list), 1)
    for field, bucket in buckets.items():
        counts = bucket["counts"]
        if len(counts) < 2:
            continue
        value_counts = [
            {"value": value, "count": count}
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        representatives = {
            value: sorted(ids)[:3]
            for value, ids in sorted(bucket["representatives"].items())
        }
        rows.append(
            {
                "field": field,
                "value_counts": value_counts,
                "representative_results": representatives,
                "source_projects": sorted(bucket["source_projects"]),
                "tension_score": round((len(counts) - 1) * sum(counts.values()) / total, 6),
            }
        )

    rows.sort(key=lambda item: (-item["tension_score"], item["field"]))
    if limit_value is not None:
        return rows[:limit_value]
    return rows
