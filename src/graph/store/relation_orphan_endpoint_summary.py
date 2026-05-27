"""Summarize relations whose endpoints do not resolve to known units."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_RELATION_KEYS = ("relation", "relation_type", "type", "predicate")
_SOURCE_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id")
_TARGET_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id")
_ID_KEYS = ("id", "edge_id", "relation_id")


def summarize_relation_orphan_endpoints(relations: Iterable[Any], units: Iterable[Any]) -> dict[str, Any]:
    """Return deterministic counts for relations with missing source or target units."""
    unit_ids = {unit_id(unit) for unit in units if unit_id(unit)}
    total_relations = orphan_relation_count = 0
    side_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)

    for index, relation in enumerate(relations):
        total_relations += 1
        meta = metadata(relation)
        relation_type = _value(relation, meta, _RELATION_KEYS) or "unknown"
        source_id = _value(relation, meta, _SOURCE_KEYS)
        target_id = _value(relation, meta, _TARGET_KEYS)
        missing_source = not source_id or source_id not in unit_ids
        missing_target = not target_id or target_id not in unit_ids
        if not (missing_source or missing_target):
            continue
        orphan_relation_count += 1
        side = "both" if missing_source and missing_target else "source" if missing_source else "target"
        side_counts[side] += 1
        type_counts[relation_type] += 1
        relation_id = _value(relation, meta, _ID_KEYS) or str(index)
        examples[(relation_type, side)].append(relation_id)

    rows = [
        {
            "relation_type": relation_type,
            "missing_endpoint_side": side,
            "count": len(ids),
            "example_relation_ids": sorted(ids, key=sort_key)[:5],
        }
        for (relation_type, side), ids in examples.items()
    ]
    rows.sort(key=lambda row: (sort_key(row["relation_type"]), sort_key(row["missing_endpoint_side"])))
    return {
        "total_relations": total_relations,
        "orphan_relation_count": orphan_relation_count,
        "missing_endpoint_counts": [{"side": side, "count": side_counts[side]} for side in sorted(side_counts, key=sort_key)],
        "relation_type_counts": [
            {"relation_type": relation_type, "count": type_counts[relation_type]}
            for relation_type in sorted(type_counts, key=sort_key)
        ],
        "rows": rows,
    }


def _value(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key))
        if value:
            return value
        value = field_value(meta.get(key))
        if value:
            return value
    return ""
