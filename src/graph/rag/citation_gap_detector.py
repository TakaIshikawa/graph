"""Detect missing citation metadata in RAG result-like records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()

_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "source": ("source", "source_name", "source_project", "publisher", "domain"),
    "url": ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri"),
    "author": ("author", "authors", "creator", "byline"),
    "date": ("date", "published_at", "publication_date", "updated_at", "created_at", "timestamp"),
}
_FIELD_WEIGHTS = {"source": 2, "url": 3, "author": 1, "date": 1}


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    citation = _field_value(payload, "citation")
    if isinstance(citation, Mapping):
        citation_value = citation.get(key, _MISSING)
        if citation_value is not _MISSING and citation_value is not None:
            return citation_value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            metadata_value = unit_metadata.get(key, _MISSING)
            if metadata_value is not _MISSING and metadata_value is not None:
                return metadata_value

    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _has_usable_value(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, Mapping):
        return any(_has_usable_value(nested) for nested in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_usable_value(nested) for nested in value)
    return _string(value) is not None


def _has_field(result: Any, field: str) -> bool:
    return any(_has_usable_value(_value(result, key)) for key in _REQUIRED_FIELDS[field])


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "result_id", "section_id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _title(result: Any) -> str | None:
    for key in ("title", "heading", "section_title"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return None


def _severity(missing_fields: list[str]) -> str:
    weight = sum(_FIELD_WEIGHTS[field] for field in missing_fields)
    if weight >= 5:
        return "high"
    if weight >= 3:
        return "medium"
    return "low"


def detect_citation_gaps(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return deterministic records for results lacking usable citation metadata."""
    gaps: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        missing_fields = [field for field in _REQUIRED_FIELDS if not _has_field(result, field)]
        if not missing_fields:
            continue

        result_id = _result_id(result, index)
        title = _title(result)
        gaps.append(
            {
                "result_id": result_id,
                "title": title,
                "missing_fields": missing_fields,
                "missing_count": len(missing_fields),
                "severity": _severity(missing_fields),
            }
        )

    gaps.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}[str(item["severity"])],
            -int(item["missing_count"]),
            str(item["title"] or "").casefold(),
            str(item["result_id"]).casefold(),
        )
    )
    return gaps
