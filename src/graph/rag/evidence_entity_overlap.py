"""Analyze entity overlap between query, claims, and evidence snippets."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z][\w-]*")
_CAP_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+){0,3})\b")
_VALUE_KEYS = ("name", "title", "label", "value", "entity", "tag")
_STOP_ENTITIES = {"a", "an", "the", "how", "tell", "what", "when", "where", "why", "who"}


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str):
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            if (text := _string(value)):
                return text
    return None


def _id(result: Any, index: int) -> str:
    return _first(result, ("id", "unit_id", "source_id")) or f"result-{index + 1}"


def _iter_strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        for key in _VALUE_KEYS:
            if (text := _string(value.get(key, _MISSING))):
                return [text]
        return []
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        values: set[str] = set()
        for item in value:
            values.update(_iter_strings(item))
        return sorted(values)
    text = _string(value)
    return [] if text is None else [text]


def _entities_from_text(text: str) -> set[str]:
    values = set(_EMAIL_RE.findall(text)) | set(_HASHTAG_RE.findall(text))
    hashtag_words = {value[1:].casefold() for value in values if value.startswith("#")}
    for match in _CAP_RE.findall(text):
        key = match.casefold()
        if key not in _STOP_ENTITIES and key not in hashtag_words:
            values.add(match.strip())
    return values


def _normalize(entity: str) -> str:
    return entity.casefold()


def _display(entities: Iterable[str]) -> list[str]:
    by_key: dict[str, str] = {}
    for entity in entities:
        by_key.setdefault(_normalize(entity), entity)
    return [by_key[key] for key in sorted(by_key)]


def _result_entities(result: Any) -> set[str]:
    texts = [
        text
        for key in ("content", "text", "snippet", "title")
        for value in _candidate_values(result, key)
        if (text := _string(value)) is not None
    ]
    entities: set[str] = set()
    for text in texts:
        entities.update(_entities_from_text(text))
    for key in ("entity", "entities", "tags"):
        for value in _candidate_values(result, key):
            entities.update(_iter_strings(value))
    return entities


def analyze_evidence_entity_overlap(query: str, results: Iterable[Any], *, claims: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return per-result entity overlap rows for query and evidence."""
    query_entities = _entities_from_text(query)
    if claims is not None:
        for claim in claims:
            text = _first(claim, ("claim", "text", "content")) or ""
            query_entities.update(_entities_from_text(text))
    query_keys = {_normalize(entity) for entity in query_entities}
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        evidence_entities = _result_entities(result)
        evidence_keys = {_normalize(entity) for entity in evidence_entities}
        shared = query_keys & evidence_keys
        missing = query_keys - evidence_keys
        rows.append(
            {
                "result_id": _id(result, index),
                "query_entities": _display(query_entities),
                "evidence_entities": _display(evidence_entities),
                "shared_entities": _display(entity for entity in query_entities if _normalize(entity) in shared),
                "missing_query_entities": _display(entity for entity in query_entities if _normalize(entity) in missing),
                "overlap_score": round(len(shared) / max(len(query_keys), 1), 6),
            }
        )
    rows.sort(key=lambda row: row["result_id"])
    return {"result_count": len(rows), "rows": rows, "summary": {"query_entity_count": len(query_keys)}}
