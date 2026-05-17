"""Summarize missing metadata fields in retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_DEFAULT_REQUIRED_FIELDS = ("title", "source", "url", "author", "published_date", "tags", "excerpt")
_FIELD_KEYS = {
    "title": ("title",),
    "source": ("source", "source_name", "source_project", "publisher", "domain"),
    "url": ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri"),
    "author": ("author", "authors", "creator", "byline"),
    "published_date": ("published_date", "published_at", "publication_date", "date", "created_at", "updated_at"),
    "tags": ("tags", "keywords", "keyphrases"),
    "excerpt": ("excerpt", "snippet", "summary", "content", "text"),
}
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")


def summarize_result_metadata_gaps(
    results: Iterable[Any],
    required_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return aggregate and per-result missing metadata fields."""
    fields = _required_fields(required_fields)
    result_list = list(results)
    missing_count_by_field: Counter[str] = Counter({field: 0 for field in fields})
    result_gaps = []

    for index, result in enumerate(result_list):
        missing_fields = [field for field in fields if not _has_field(result, field)]
        for field in missing_fields:
            missing_count_by_field[field] += 1
        result_gaps.append({"result_id": _result_id(result, index), "missing_fields": missing_fields})

    return {
        "total_results": len(result_list),
        "result_gaps": result_gaps,
        "missing_count_by_field": dict(sorted(missing_count_by_field.items())),
    }


def _required_fields(required_fields: Iterable[str] | None) -> tuple[str, ...]:
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
    yield _field_value(payload, key)
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        yield metadata.get(key, _MISSING)
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        yield _field_value(unit, key)
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            yield unit_metadata.get(key, _MISSING)


def _has_value(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_value(item) for item in value)
    return True


def _has_field(result: Any, field: str) -> bool:
    return any(_has_value(value) for key in _FIELD_KEYS.get(field, (field,)) for value in _candidate_values(result, key))


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return f"result-{index + 1}"
