"""Summarize simple directed relation cycles."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_SOURCE_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id")
_TARGET_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id")
_TYPE_KEYS = ("relation", "relation_type", "type", "predicate")


def summarize_relation_cycles(relations: Iterable[Any], *, max_depth: int = 4, sample_limit: int = 5) -> dict[str, Any]:
    if not isinstance(max_depth, int) or max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")

    edges = [_edge(relation) for relation in relations]
    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, target, relation_type in edges:
        if source and target:
            graph[source].append((target, relation_type))
    for outgoing in graph.values():
        outgoing.sort(key=lambda item: (sort_key(item[0]), sort_key(item[1])))

    cycles: dict[tuple[str, ...], list[str]] = {}
    for start in sorted(graph, key=sort_key):
        _walk(graph, start, start, [], [], max_depth, cycles)

    ordered = sorted(cycles.items(), key=lambda item: (len(item[0]), tuple(sort_key(node) for node in item[0])))
    sampled = ordered[: max(0, sample_limit)]
    type_counts: Counter[str] = Counter(relation_type for _, types in sampled for relation_type in types)
    nodes = {node for cycle, _ in ordered for node in cycle}
    return {
        "cycle_count": len(ordered),
        "node_count_in_cycles": len(nodes),
        "relation_type_counts": dict(sorted(type_counts.items(), key=lambda item: sort_key(item[0]))),
        "cycle_samples": [
            {"nodes": list(cycle), "relation_types": types}
            for cycle, types in sampled
        ],
    }


def _walk(
    graph: Mapping[str, list[tuple[str, str]]],
    start: str,
    current: str,
    path: list[str],
    types: list[str],
    max_depth: int,
    cycles: dict[tuple[str, ...], list[str]],
) -> None:
    path = [*path, current]
    if len(path) > max_depth:
        return
    for target, relation_type in graph.get(current, []):
        if target == start and len(path) >= 2:
            key = _canonical(path)
            cycles.setdefault(key, [*types, relation_type])
        elif target not in path:
            _walk(graph, start, target, path, [*types, relation_type], max_depth, cycles)


def _canonical(nodes: list[str]) -> tuple[str, ...]:
    rotations = [tuple(nodes[index:] + nodes[:index]) for index in range(len(nodes))]
    return min(rotations, key=lambda row: tuple(sort_key(node) for node in row))


def _edge(relation: Any) -> tuple[str, str, str]:
    meta = metadata(relation)
    return (_first(relation, meta, _SOURCE_KEYS), _first(relation, meta, _TARGET_KEYS), _first(relation, meta, _TYPE_KEYS) or "unknown")


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
