"""Classify retrieved RAG/search results by personal or external scope."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()
_ID_KEYS = ("id", "unit_id", "source_id")
_PERSONAL_PROJECTS = {"notes", "tasks", "calendar", "local", "personal", "max", "activitywatch"}
_PERSONAL_TYPES = {"note", "task", "calendar", "calendar_event", "event", "todo", "file"}
_EXTERNAL_FIELDS = ("doi", "isbn", "isbn10", "isbn13", "publisher", "journal", "publication", "source_name")


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


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, _ID_KEYS) or f"result-{index + 1}"


def _has_value(result: Any, keys: tuple[str, ...]) -> bool:
    return _first(result, keys) is not None


def _url_signals(result: Any) -> tuple[list[str], list[str]]:
    personal: list[str] = []
    external: list[str] = []
    for key in ("url", "source_url", "canonical_url", "uri", "path"):
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is None:
                continue
            parsed = urlsplit(text)
            if parsed.scheme in {"http", "https"} and parsed.hostname:
                external.append("http-domain")
            if parsed.scheme == "file" or text.startswith(("/", "~")):
                personal.append("file-url")
    return personal, external


def _classify(result: Any) -> tuple[str, list[str]]:
    personal: list[str] = []
    external: list[str] = []
    project = (_first(result, ("source_project",)) or "").casefold()
    if project in _PERSONAL_PROJECTS:
        personal.append("personal-source-project")
    entity_type = (_first(result, ("source_entity_type", "content_type", "type")) or "").casefold()
    if entity_type in _PERSONAL_TYPES:
        personal.append("personal-entity-type")
    url_personal, url_external = _url_signals(result)
    personal.extend(url_personal)
    external.extend(url_external)
    if _has_value(result, _EXTERNAL_FIELDS):
        external.append("publisher-or-identifier")

    if personal and external:
        return "mixed", sorted(set(personal + external))
    if personal:
        return "personal", sorted(set(personal))
    if external:
        return "external", sorted(set(external))
    return "unknown", []


def classify_result_scope(results: Iterable[Any]) -> dict[str, Any]:
    """Classify retrieved results as personal, external, mixed, or unknown."""
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, result in enumerate(results):
        scope, signals = _classify(result)
        counts[scope] += 1
        rows.append({"result_id": _result_id(result, index), "scope": scope, "signals": signals})

    total = len(rows)
    scope_counts = {scope: counts.get(scope, 0) for scope in ("personal", "external", "mixed", "unknown")}
    scope_percentages = {
        scope: round((count / total) * 100, 1) if total else 0.0
        for scope, count in scope_counts.items()
    }
    return {"scope_counts": scope_counts, "scope_percentages": scope_percentages, "results": rows}
