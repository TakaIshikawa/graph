"""Summarize relation self loops."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, sort_key

_RELATION_KEYS = ("relation", "relation_type", "type", "predicate")
_SOURCE_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id", "source")
_TARGET_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id", "target")
_SOURCE_METADATA_KEYS = ("metadata_source", "source", "edge_source", "source_project")


def summarize_relation_self_loops(relations: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    """Return deterministic counts for relations whose source and target ids match."""

    total_relations = self_loop_count = missing_endpoint_count = 0
    type_counts: Counter[str] = Counter()
    metadata_source_counts: Counter[str] = Counter()
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "sample_relations": []}
    )

    for index, relation in enumerate(relations):
        total_relations += 1
        meta = metadata(relation)
        relation_type = _value(relation, meta, _RELATION_KEYS) or "unknown"
        source_id = _value(relation, meta, _SOURCE_KEYS)
        target_id = _value(relation, meta, _TARGET_KEYS)
        if not source_id or not target_id:
            missing_endpoint_count += 1
            continue
        if source_id != target_id:
            continue

        self_loop_count += 1
        metadata_source = _value(relation, meta, _SOURCE_METADATA_KEYS) or "unknown"
        type_counts[relation_type] += 1
        metadata_source_counts[metadata_source] += 1
        group = groups[(relation_type, metadata_source)]
        group["count"] += 1
        if len(group["sample_relations"]) < sample_limit:
            relation_id = edge_id(relation) or str(index)
            group["sample_relations"].append(
                {"relation_id": relation_id, "endpoint_id": source_id}
            )

    rows = []
    for relation_type, metadata_source in sorted(
        groups, key=lambda key: (sort_key(key[0]), sort_key(key[1]))
    ):
        group = groups[(relation_type, metadata_source)]
        rows.append(
            {
                "relation_type": relation_type,
                "metadata_source": metadata_source,
                "count": group["count"],
                "sample_relations": group["sample_relations"],
            }
        )

    return {
        "total_relations": total_relations,
        "self_loop_count": self_loop_count,
        "missing_endpoint_count": missing_endpoint_count,
        "relation_type_counts": [
            {"relation_type": relation_type, "count": type_counts[relation_type]}
            for relation_type in sorted(type_counts, key=sort_key)
        ],
        "metadata_source_counts": [
            {"metadata_source": source, "count": metadata_source_counts[source]}
            for source in sorted(metadata_source_counts, key=sort_key)
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
