"""Summarize relation type frequencies."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, sort_key

_TYPE_KEYS = ("relation_type", "relation", "type", "predicate")
_MISSING = "missing"


def summarize_relation_type_frequency(relations: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    total = 0
    for index, relation in enumerate(relations):
        total += 1
        meta = metadata(relation)
        relation_type = _first(relation, meta, _TYPE_KEYS) or _MISSING
        counts[relation_type] += 1
        relation_id = edge_id(relation) or str(index)
        if len(samples[relation_type]) < max(0, sample_limit):
            samples[relation_type].append(relation_id)
    rows = [
        {"relation_type": relation_type, "count": count, "sample_relation_ids": sorted(samples[relation_type], key=sort_key)}
        for relation_type, count in sorted(counts.items(), key=lambda item: (-item[1], sort_key(item[0])))
    ]
    return {"total_relations": total, "missing_type_count": counts[_MISSING], "relation_type_counts": rows}


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
