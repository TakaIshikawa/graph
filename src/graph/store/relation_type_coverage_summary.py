"""Summarize relation type coverage across graph relations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_RELATION_KEYS = ("relation", "relation_type", "type", "predicate")
_SOURCE_UNIT_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id")
_TARGET_UNIT_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id")
_SOURCE_KEYS = ("source", "edge_source", "source_project")
_EVIDENCE_KEYS = ("evidence", "evidence_ids", "supporting_evidence")
_WEIGHT_KEYS = ("weight", "score", "confidence")


def summarize_relation_type_coverage(relations: Iterable[Any]) -> dict[str, Any]:
    """Group relations by type and summarize endpoint, evidence, and weight coverage."""

    groups: dict[str | None, dict[str, Any]] = {}
    total_edges = 0
    for relation in relations:
        total_edges += 1
        metadata = _metadata(relation)
        relation_type = _string(_first(relation, metadata, _RELATION_KEYS))
        group = groups.setdefault(
            relation_type,
            {
                "relation_type": relation_type,
                "edge_count": 0,
                "source_units": set(),
                "target_units": set(),
                "sources": set(),
                "missing_evidence_count": 0,
                "weights": [],
            },
        )
        group["edge_count"] += 1
        source_unit = _string(_first(relation, metadata, _SOURCE_UNIT_KEYS))
        target_unit = _string(_first(relation, metadata, _TARGET_UNIT_KEYS))
        source = _string(_first(relation, metadata, _SOURCE_KEYS))
        if source_unit is not None:
            group["source_units"].add(source_unit)
        if target_unit is not None:
            group["target_units"].add(target_unit)
        if source is not None:
            group["sources"].add(source)
        if not _evidence(relation, metadata):
            group["missing_evidence_count"] += 1
        weight = _number(_first(relation, metadata, _WEIGHT_KEYS))
        if weight is not None:
            group["weights"].append(weight)

    rows = []
    for relation_type in sorted(groups, key=lambda value: value or ""):
        group = groups[relation_type]
        weights = group["weights"]
        rows.append(
            {
                "relation_type": relation_type,
                "edge_count": group["edge_count"],
                "unique_source_unit_count": len(group["source_units"]),
                "unique_target_unit_count": len(group["target_units"]),
                "distinct_source_count": len(group["sources"]),
                "missing_evidence_count": group["missing_evidence_count"],
                "average_weight": sum(weights) / len(weights) if weights else None,
            }
        )
    return {"rows": rows, "row_count": len(rows), "edge_count": total_edges}


def _evidence(item: Any, metadata: Mapping[str, Any]) -> list[Any]:
    for key in _EVIDENCE_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return _as_list(value)
        value = metadata.get(key)
        if value not in (None, ""):
            return _as_list(value)
    return []


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    return [value] if value not in (None, "") else []


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _first(item: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get(item, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
