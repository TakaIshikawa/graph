"""Analyze tag coverage across retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")
_TAG_KEYS = ("tags", "keywords", "keyphrases")
_TAG_VALUE_KEYS = ("tag", "keyword", "term", "phrase", "key", "value")


def analyze_result_tag_coverage(results: Iterable[Any]) -> dict[str, Any]:
    """Return tag counts, untagged ids, dominant tags, rare tags, and coverage ratio."""
    result_list = list(results)
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    untagged_result_ids = []

    for index, result in enumerate(result_list):
        normalized_tags = []
        for tag in _tags(result):
            normalized = tag.casefold()
            if normalized not in labels:
                labels[normalized] = tag
            if normalized not in normalized_tags:
                normalized_tags.append(normalized)
        if normalized_tags:
            counts.update(normalized_tags)
        else:
            untagged_result_ids.append(_result_id(result, index))

    tag_counts = {labels[tag]: counts[tag] for tag in sorted(counts, key=lambda tag: (labels[tag].casefold(), labels[tag]))}
    max_count = max(counts.values(), default=0)
    dominant_tags = [labels[tag] for tag, count in sorted(counts.items()) if count == max_count and max_count > 0]
    rare_tags = [labels[tag] for tag, count in sorted(counts.items()) if count == 1]
    covered = len(result_list) - len(untagged_result_ids)
    return {
        "tag_counts": tag_counts,
        "untagged_result_ids": untagged_result_ids,
        "dominant_tags": dominant_tags,
        "rare_tags": rare_tags,
        "coverage_ratio": round(covered / len(result_list), 3) if result_list else 0.0,
    }


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


def _tags_from_value(value: Any) -> Iterable[str]:
    if value is _MISSING or value is None:
        return
    if isinstance(value, Mapping):
        for key in _TAG_VALUE_KEYS:
            text = _string(value.get(key, _MISSING))
            if text is not None:
                yield text
                return
        for key in value:
            text = _string(key)
            if text is not None:
                yield text
        return
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            yield from _tags_from_value(item)
        return
    text = _string(value)
    if text is not None:
        yield text


def _tags(result: Any) -> list[str]:
    tags: list[str] = []
    for key in _TAG_KEYS:
        for tag in _tags_from_value(_value(result, key)):
            if tag.casefold() not in {existing.casefold() for existing in tags}:
                tags.append(tag)
    return tags


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        text = _string(_value(result, key))
        if text is not None:
            return text
    return f"result-{index + 1}"
