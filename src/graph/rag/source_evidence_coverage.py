"""Analyze evidence coverage grouped by RAG result source."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_MISSING_SOURCE = "__missing_source__"
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")
_SOURCE_KEYS = ("source", "source_name", "source_project", "publisher", "domain")
_EVIDENCE_KEYS = ("evidence", "evidence_items", "snippets", "quotes", "snippet", "excerpt", "content")
_CITATION_KEYS = ("citation", "citations", "references", "url", "source_url", "canonical_url", "doi")


def analyze_source_evidence_coverage(results: Iterable[Any]) -> dict[str, Any]:
    """Return per-source evidence and citation coverage for retrieved results."""
    groups: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(results):
        source_key = _source_key(result)
        group = groups.setdefault(
            source_key,
            {"source_key": source_key, "result_count": 0, "evidence_count": 0, "cited_count": 0, "result_ids": []},
        )
        group["result_count"] += 1
        group["evidence_count"] += _evidence_count(result)
        group["cited_count"] += int(any(_has_value(_value(result, key)) for key in _CITATION_KEYS))
        group["result_ids"].append(_result_id(result, index))

    sources = sorted(groups.values(), key=lambda item: (str(item["source_key"]).casefold(), str(item["source_key"])))
    return {"source_count": len(sources), "sources": sources}


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


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key, _MISSING)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _has_value(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_value(item) for item in value)
    return True


def _count(value: Any) -> int:
    if not _has_value(value):
        return 0
    if isinstance(value, Mapping):
        return sum(1 for item in value.values() if _has_value(item))
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return sum(1 for item in value if _has_value(item))
    return 1


def _source_key(result: Any) -> str:
    for key in _SOURCE_KEYS:
        text = _string(_value(result, key))
        if text is not None:
            return text
    return _MISSING_SOURCE


def _evidence_count(result: Any) -> int:
    return sum(_count(_value(result, key)) for key in _EVIDENCE_KEYS)


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        text = _string(_value(result, key))
        if text is not None:
            return text
    return f"result-{index + 1}"
