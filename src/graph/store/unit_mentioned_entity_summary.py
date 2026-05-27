"""Mentioned entity summary for store units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

ENTITY_KEYS = ("entities", "mentions", "people", "organizations", "topics")


def summarize_unit_mentioned_entities(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        grouped[_source(unit)].append(unit)

    rows: list[dict[str, Any]] = []
    for source in sorted(grouped, key=_sort_key):
        counts: Counter[str] = Counter()
        per_unit: list[int] = []
        for unit in grouped[source]:
            entities = _unit_entities(unit)
            per_unit.append(len(entities))
            counts.update(entities)
        unit_count = len(grouped[source])
        total_mentions = sum(per_unit)
        rows.append(
            {
                "source": source,
                "source_project": source,
                "unit_count": unit_count,
                "entity_unit_count": sum(1 for count in per_unit if count > 0),
                "total_entity_mentions": total_mentions,
                "distinct_entity_count": len(counts),
                "average_entities_per_unit": round(total_mentions / unit_count, 2) if unit_count else 0.0,
                "top_entities": [{"entity": entity, "count": count} for entity, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))],
            }
        )
    return {"rows": rows, "source_summaries": rows, "total_units": sum(row["unit_count"] for row in rows)}


def _unit_entities(unit: Mapping[str, Any] | object) -> list[str]:
    metadata = _metadata(unit)
    entities: list[str] = []
    for key in ENTITY_KEYS:
        entities.extend(_entity_values(_get(unit, key)))
        entities.extend(_entity_values(metadata.get(key)))
    return [entity for entity in (_normalize(value) for value in entities) if entity]


def _entity_values(value: object) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("name", "text", "id"):
            text = _text(value.get(key))
            if text:
                return [text]
        return []
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_entity_values(item))
        return values
    text = _text(value)
    if not text:
        return []
    return [_text(part) for part in text.split(",")] if "," in text else [text]


def _normalize(value: object) -> str:
    return " ".join(_text(value).split())


def _source(unit: Mapping[str, Any] | object) -> str:
    metadata = _metadata(unit)
    return _text(_get(unit, "source_project")) or _text(_get(unit, "source")) or _text(metadata.get("source")) or "unknown"


def _metadata(value: Mapping[str, Any] | object) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
