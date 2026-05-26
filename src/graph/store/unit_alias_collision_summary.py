"""Summarize unit alias collisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_ALIAS_KEYS = ("aliases", "alias", "slug", "title_slug", "canonical_name")


def summarize_unit_alias_collisions(units: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"aliases": set(), "unit_ids": set(), "sources": set(), "entity_types": set()})
    total_units = 0
    for unit in units:
        total_units += 1
        unit_id = _text(_get(unit, "id") or _get(unit, "unit_id"))
        for alias in set(_aliases(unit)):
            normalized = _normalize(alias)
            if not normalized:
                continue
            group = grouped[normalized]
            group["aliases"].add(alias)
            group["unit_ids"].add(unit_id)
            source = _text(_get(unit, "source_project") or _metadata(unit).get("source") or _metadata(unit).get("source_project"))
            entity_type = _text(_get(unit, "source_entity_type") or _get(unit, "entity_type") or _metadata(unit).get("entity_type"))
            if source:
                group["sources"].add(source)
            if entity_type:
                group["entity_types"].add(entity_type)
    rows = []
    for normalized, group in grouped.items():
        if len(group["unit_ids"]) <= 1:
            continue
        rows.append(
            {
                "alias": sorted(group["aliases"], key=_sort_key)[0],
                "normalized_alias": normalized,
                "unit_count": len(group["unit_ids"]),
                "unit_ids": sorted(group["unit_ids"], key=_sort_key),
                "sources": sorted(group["sources"], key=_sort_key),
                "entity_types": sorted(group["entity_types"], key=_sort_key),
            }
        )
    rows.sort(key=lambda row: (_sort_key(row["normalized_alias"]), row["unit_ids"]))
    return {"total_units": total_units, "collision_count": len(rows), "rows": rows, "alias_collisions": rows}


def _aliases(unit: Any) -> list[str]:
    meta = _metadata(unit)
    found: list[str] = []
    for key in _ALIAS_KEYS:
        found.extend(_flatten(_get(unit, key) or meta.get(key)))
    return [_text(value) for value in found if _text(value)]


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _normalize(value: str) -> str:
    return _text(value).casefold()


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
