"""Summarize relation metadata key frequencies by relation type."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key


def summarize_relation_metadata_key_frequency(edges: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    edge_counts: Counter[str] = Counter()
    total = 0
    for edge in edges:
        total += 1
        relation_type = _relation_type(edge)
        edge_counts[relation_type] += 1
        grouped[relation_type].update(_flatten_metadata_keys(metadata(edge)))

    relation_summaries = []
    total_counts: Counter[str] = Counter()
    for relation_type in sorted(grouped, key=sort_key):
        counts = grouped[relation_type]
        total_counts.update(counts)
        relation_summaries.append(
            {
                "relation_type": relation_type,
                "edge_count": edge_counts[relation_type],
                "metadata_key_count": len(counts),
                "keys": _rows(counts),
            }
        )

    return {"total_edges": total, "relation_summaries": relation_summaries, "top_keys": _rows(total_counts)}


def _flatten_metadata_keys(value: Mapping[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for raw_key, child in value.items():
        key = field_value(raw_key).casefold()
        if not key:
            continue
        path = f"{prefix}.{key}" if prefix else key
        keys.append(path)
        if isinstance(child, Mapping):
            keys.extend(_flatten_metadata_keys(child, path))
    return keys


def _relation_type(edge: Any) -> str:
    return field_value(get(edge, "relation_type")) or field_value(get(edge, "type")) or "unknown"


def _rows(counts: Counter[str]) -> list[dict[str, Any]]:
    return [{"key": key, "count": counts[key]} for key in sorted(counts, key=lambda key: (-counts[key], sort_key(key)))]
