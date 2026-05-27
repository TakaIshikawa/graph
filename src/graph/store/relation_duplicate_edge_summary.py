"""Summarize duplicate relation edges."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, sort_key

_SOURCE_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id", "source")
_TARGET_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id", "target")
_TYPE_KEYS = ("relation_type", "relation", "type", "predicate")


def summarize_relation_duplicate_edges(relations: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total_relations = 0
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for relation in relations:
        total_relations += 1
        meta = metadata(relation)
        groups[(_value(relation, meta, _SOURCE_KEYS), _value(relation, meta, _TARGET_KEYS), _value(relation, meta, _TYPE_KEYS) or "unknown")].append(relation)
    duplicate_groups = []
    duplicate_edge_count = 0
    for (source_id, target_id, relation_type), items in groups.items():
        if len(items) < 2:
            continue
        duplicate_edge_count += len(items)
        key_sets = {tuple(sorted(metadata(item).keys(), key=sort_key)) for item in items}
        duplicate_groups.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "count": len(items),
                "edge_ids": [edge_id(item) for item in items],
                "metadata_key_variation_count": len(key_sets),
            }
        )
    duplicate_groups.sort(key=lambda row: (-row["count"], sort_key(row["source_id"]), sort_key(row["target_id"]), sort_key(row["relation_type"])))
    return {
        "total_relations": total_relations,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_edge_count": duplicate_edge_count,
        "unique_edge_count": sum(1 for items in groups.values() if len(items) == 1),
        "groups": duplicate_groups,
        "samples": duplicate_groups[: max(0, sample_limit)],
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
