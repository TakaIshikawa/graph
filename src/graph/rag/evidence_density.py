"""Score RAG results by local evidence density signals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet", "summary")
_CITATION_KEYS = (
    "citations",
    "references",
    "citation",
    "reference",
    "url",
    "source_url",
    "canonical_url",
    "doi",
    "isbn",
)
_METADATA_FIELDS = {
    "source": ("source", "source_name", "source_project", "publisher", "domain"),
    "url": ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri"),
    "author": ("author", "authors", "creator", "byline"),
    "date": ("date", "published_at", "publication_date", "updated_at", "created_at", "timestamp"),
}
_RELATION_KEYS = ("relations", "relation_count", "edge_count", "linked_units")
_SOURCE_COUNT_KEYS = ("source_count", "sources", "source_ids")


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


def _has_signal(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, Mapping):
        return any(_has_signal(nested) for nested in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_signal(nested) for nested in value)
    return _string(value) is not None


def _count_signal(value: Any) -> int:
    if not _has_signal(value):
        return 0
    if isinstance(value, Mapping):
        return sum(1 for nested in value.values() if _has_signal(nested))
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return sum(1 for nested in value if _has_signal(nested))
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(int(value), 1)
    return 1


def _numeric_count(result: Any, keys: tuple[str, ...]) -> int:
    best = 0
    for key in keys:
        value = _value(result, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            best = max(best, max(int(value), 0))
        else:
            best = max(best, _count_signal(value))
    return best


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "result_id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _title(result: Any) -> str | None:
    return _string(_value(result, "title"))


def _word_count(result: Any) -> int:
    parts = []
    title = _string(_value(result, "title"))
    if title is not None:
        parts.append(title)
    for key in _TEXT_KEYS:
        value = _string(_value(result, key))
        if value is not None:
            parts.append(value)
    return len(TOKEN_RE.findall(" ".join(parts).casefold()))


def _citation_count(result: Any) -> int:
    return sum(_count_signal(_value(result, key)) for key in _CITATION_KEYS)


def _metadata_completeness(result: Any) -> int:
    count = 0
    for keys in _METADATA_FIELDS.values():
        if any(_has_signal(_value(result, key)) for key in keys):
            count += 1
    return count


def score_evidence_density(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return sorted evidence density records for result-like dictionaries."""
    rows: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        text_word_count = _word_count(result)
        citation_count = _citation_count(result)
        metadata_field_count = _metadata_completeness(result)
        relation_count = _numeric_count(result, _RELATION_KEYS)
        source_count = _numeric_count(result, _SOURCE_COUNT_KEYS)
        graph_context_count = relation_count + source_count
        density_score = round(
            min(text_word_count, 200) * 0.01
            + citation_count * 1.5
            + metadata_field_count
            + graph_context_count * 0.5,
            3,
        )
        rows.append(
            {
                "result_id": _result_id(result, index),
                "title": _title(result),
                "density_score": density_score,
                "text_word_count": text_word_count,
                "citation_count": citation_count,
                "metadata_field_count": metadata_field_count,
                "relation_count": relation_count,
                "source_count": source_count,
                "graph_context_count": graph_context_count,
            }
        )

    rows.sort(
        key=lambda item: (
            -float(item["density_score"]),
            -int(item["citation_count"]),
            -int(item["metadata_field_count"]),
            str(item["title"] or "").casefold(),
            str(item["result_id"]).casefold(),
        )
    )
    return rows
