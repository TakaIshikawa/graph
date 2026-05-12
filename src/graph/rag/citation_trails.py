"""Build compact graph citation trails for RAG/search results."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

_CITATION_METADATA_KEYS = (
    "citation",
    "citations",
    "citation_key",
    "doi",
    "url",
    "source_url",
    "canonical_url",
    "reference",
    "references",
)
_MISSING = object()


def _field_value(item: Any, key: str) -> Any:
    if item is None or item is _MISSING:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(item: Any) -> Any:
    if isinstance(item, tuple) and item:
        return item[0]
    if isinstance(item, Mapping) and "unit" in item:
        return item["unit"]
    return item


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _unit_id(item: Any) -> str | None:
    item = _payload(item)
    for key in ("id", "unit_id"):
        value = _string_value(_field_value(item, key))
        if value is not None:
            return value
    return None


def _metadata(item: Any) -> Mapping:
    item = _payload(item)
    metadata = _field_value(item, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _citation_metadata(item: Any) -> dict[str, Any]:
    metadata = _metadata(item)
    citation = {
        key: metadata[key]
        for key in _CITATION_METADATA_KEYS
        if key in metadata and metadata[key] not in (None, "", [], {})
    }
    for key in ("source_id", "source_entity_type"):
        value = _field_value(_payload(item), key)
        if value is not _MISSING and value not in (None, ""):
            citation[key] = str(value)
    return citation


def _unit_summary(item: Any, unit_id: str | None = None) -> dict[str, Any]:
    item = _payload(item)
    found_id = unit_id or _unit_id(item) or ""
    return {
        "id": found_id,
        "title": _string_value(_field_value(item, "title")) or found_id,
        "source_project": _string_value(_field_value(item, "source_project")),
        "citation": _citation_metadata(item),
    }


def _edge_endpoint(edge: Any, *keys: str) -> str | None:
    for key in keys:
        value = _string_value(_field_value(edge, key))
        if value is not None:
            return value
    return None


def _edge_unit(edge: Any, *keys: str) -> Any:
    for key in keys:
        value = _field_value(edge, key)
        if value is not _MISSING and value is not None:
            return value
    return None


def _edge_relation(edge: Any) -> str:
    value = _field_value(edge, "relation")
    text = _string_value(value)
    return text or "related"


def _edge_record(edge: Any, from_id: str, to_id: str, units: dict[str, Any]) -> dict[str, Any]:
    return {
        "from": _unit_summary(units.get(from_id), from_id),
        "relation": _edge_relation(edge),
        "to": _unit_summary(units.get(to_id), to_id),
    }


def build_citation_trails(
    results: Iterable[Any],
    edges: Iterable[Any],
    *,
    max_depth: int = 2,
    max_trails: int = 20,
) -> list[dict[str, Any]]:
    """Build deterministic citation trails from result/unit records and edges."""
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer.")
    if not isinstance(max_trails, int) or isinstance(max_trails, bool) or max_trails < 0:
        raise ValueError("max_trails must be a non-negative integer.")
    if max_trails == 0:
        return []

    result_items = list(results)
    edge_items = list(edges)
    units: dict[str, Any] = {}
    root_ids: list[str] = []
    for result in result_items:
        unit_id = _unit_id(result)
        if unit_id is None:
            continue
        units.setdefault(unit_id, _payload(result))
        if unit_id not in root_ids:
            root_ids.append(unit_id)

    adjacency: dict[str, list[tuple[str, str, Any]]] = {}
    for edge in edge_items:
        from_id = _edge_endpoint(edge, "from_unit_id", "from_id", "source", "from")
        to_id = _edge_endpoint(edge, "to_unit_id", "to_id", "target", "to")
        from_unit = _edge_unit(edge, "from_unit", "source_unit")
        to_unit = _edge_unit(edge, "to_unit", "target_unit")
        if from_id is None and from_unit is not None:
            from_id = _unit_id(from_unit)
        if to_id is None and to_unit is not None:
            to_id = _unit_id(to_unit)
        if from_id is None or to_id is None or from_id == to_id:
            continue
        if from_unit is not None:
            units.setdefault(from_id, from_unit)
        if to_unit is not None:
            units.setdefault(to_id, to_unit)
        adjacency.setdefault(from_id, []).append(("outbound", to_id, edge))
        adjacency.setdefault(to_id, []).append(("inbound", from_id, edge))

    for unit_id in adjacency:
        adjacency[unit_id].sort(
            key=lambda item: (_edge_relation(item[2]), item[0], item[1])
        )

    trails: list[dict[str, Any]] = []
    for root_id in sorted(root_ids):
        queue = deque([(root_id, [], {root_id})])
        while queue:
            current_id, path, seen = queue.popleft()
            if len(path) >= max_depth:
                continue
            for direction, next_id, edge in adjacency.get(current_id, []):
                if next_id in seen:
                    continue
                from_id, to_id = (
                    (current_id, next_id) if direction == "outbound" else (next_id, current_id)
                )
                next_path = [*path, _edge_record(edge, from_id, to_id, units)]
                trails.append(
                    {
                        "root": _unit_summary(units.get(root_id), root_id),
                        "depth": len(next_path),
                        "path": next_path,
                    }
                )
                queue.append((next_id, next_path, {*seen, next_id}))

    trails.sort(
        key=lambda trail: (
            trail["depth"],
            trail["root"]["id"],
            [
                (
                    step["relation"],
                    step["from"]["id"],
                    step["to"]["id"],
                )
                for step in trail["path"]
            ],
        )
    )
    return trails[:max_trails]
