"""Analyze provenance completeness for retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_DEFAULT_REQUIRED_FIELDS = (
    "source_id",
    "source_project",
    "source_entity_type",
    "citation",
)
_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "source_id": ("source_id",),
    "source_project": ("source_project", "source", "source_name", "project"),
    "source_entity_type": ("source_entity_type", "entity_type", "type", "kind"),
    "citation": (
        "citation",
        "citations",
        "citation_url",
        "url",
        "source_url",
        "canonical_url",
        "external_url",
        "link",
        "permalink",
        "uri",
        "doi",
    ),
}
_IDENTIFIER_KEYS = ("id", "unit_id", "source_id")


def analyze_result_provenance_completeness(
    results: Iterable[Any],
    required_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return aggregate and per-result provenance gaps for RAG results."""
    fields = _validate_required_fields(required_fields)
    result_list = list(results)
    missing_counts: Counter[str] = Counter({field: 0 for field in fields})
    per_result_gaps: list[dict[str, Any]] = []
    complete_count = 0

    for index, result in enumerate(result_list):
        missing = [
            field
            for field in fields
            if not _has_provenance_value(result, field)
        ]
        if missing:
            for field in missing:
                missing_counts[field] += 1
            per_result_gaps.append(
                {
                    "result_id": _result_id(result, index),
                    "missing_fields": missing,
                }
            )
        else:
            complete_count += 1

    total = len(result_list)
    incomplete_count = total - complete_count
    return {
        "total_results": total,
        "complete_result_count": complete_count,
        "incomplete_result_count": incomplete_count,
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "per_result_gaps": per_result_gaps,
        "completeness_percent": round((complete_count / total) * 100, 1) if total else 0.0,
    }


def _validate_required_fields(required_fields: Iterable[str] | None) -> tuple[str, ...]:
    if required_fields is None:
        return _DEFAULT_REQUIRED_FIELDS

    fields: list[str] = []
    for field in required_fields:
        if not isinstance(field, str) or not field.strip():
            raise ValueError("required_fields must contain non-empty strings")
        normalized = field.strip().casefold()
        if normalized not in fields:
            fields.append(normalized)
    if not fields:
        raise ValueError("required_fields must contain at least one field")
    return tuple(fields)


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
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _field_keys(field: str) -> tuple[str, ...]:
    return _FIELD_KEYS.get(field, (field,))


def _has_provenance_value(result: Any, field: str) -> bool:
    return any(_has_value(value) for key in _field_keys(field) for value in _candidate_values(result, key))


def _has_value(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, bool):
        return True
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_value(item) for item in value)
    return True


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in _IDENTIFIER_KEYS:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return f"result-{index + 1}"
