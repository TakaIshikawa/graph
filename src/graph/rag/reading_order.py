"""Deterministic graph-aware reading order planning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge, KnowledgeUnit

PREREQUISITE_RELATIONS = frozenset(
    {
        EdgeRelation.BUILDS_ON.value,
        EdgeRelation.DERIVES_FROM.value,
        EdgeRelation.REFERENCES.value,
        EdgeRelation.CONTAINS.value,
    }
)


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    return limit


def _relation_value(relation: EdgeRelation | str) -> str:
    return relation.value if isinstance(relation, EdgeRelation) else str(relation)


def _unit_payload(unit: KnowledgeUnit, reason: str) -> dict[str, Any]:
    return {
        "id": unit.id,
        "source_project": str(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content_type": str(unit.content_type),
        "reason": reason,
    }


def _component_for_seed(
    unit_ids: set[str], edges: Iterable[KnowledgeEdge], seed_unit_id: str | None
) -> tuple[set[str], bool]:
    if seed_unit_id is None:
        return set(unit_ids), False
    if seed_unit_id not in unit_ids:
        return set(), False

    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        left = str(edge.from_unit_id)
        right = str(edge.to_unit_id)
        if left not in unit_ids or right not in unit_ids:
            continue
        neighbors[left].add(right)
        neighbors[right].add(left)

    seen = {seed_unit_id}
    stack = [seed_unit_id]
    while stack:
        current = stack.pop()
        for neighbor in sorted(neighbors[current], reverse=True):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen, True


def _add_ordering_edge(
    edge: KnowledgeEdge,
    relevant_ids: set[str],
    outgoing: dict[str, set[str]],
    incoming: dict[str, set[str]],
) -> None:
    relation = _relation_value(edge.relation)
    if relation not in PREREQUISITE_RELATIONS:
        return

    from_unit_id = str(edge.from_unit_id)
    to_unit_id = str(edge.to_unit_id)
    if from_unit_id not in relevant_ids or to_unit_id not in relevant_ids:
        return
    if from_unit_id == to_unit_id:
        incoming[from_unit_id].add(from_unit_id)
        outgoing[from_unit_id].add(from_unit_id)
        return

    if relation == EdgeRelation.CONTAINS.value:
        prerequisite_id = from_unit_id
        dependent_id = to_unit_id
    else:
        prerequisite_id = to_unit_id
        dependent_id = from_unit_id

    outgoing[prerequisite_id].add(dependent_id)
    incoming[dependent_id].add(prerequisite_id)


def _topological_order(
    relevant_ids: set[str], edges: Iterable[KnowledgeEdge]
) -> tuple[list[str], set[str]]:
    outgoing = {unit_id: set() for unit_id in relevant_ids}
    incoming = {unit_id: set() for unit_id in relevant_ids}
    for edge in edges:
        _add_ordering_edge(edge, relevant_ids, outgoing, incoming)

    remaining = set(relevant_ids)
    ready = {unit_id for unit_id in remaining if not incoming[unit_id]}
    order: list[str] = []
    cycle_fallback_ids: set[str] = set()

    while remaining:
        if ready:
            current = min(ready)
            ready.remove(current)
        else:
            current = min(remaining)
            cycle_fallback_ids.add(current)

        if current not in remaining:
            continue

        remaining.remove(current)
        order.append(current)
        for dependent_id in sorted(outgoing[current]):
            if dependent_id not in remaining:
                continue
            incoming[dependent_id].discard(current)
            if not incoming[dependent_id]:
                ready.add(dependent_id)

    return order, cycle_fallback_ids


def _reason_for_unit(
    unit_id: str,
    *,
    seed_unit_id: str | None,
    seed_found: bool,
    cycle_fallback_ids: set[str],
) -> str:
    if unit_id in cycle_fallback_ids:
        return "cycle_fallback"
    if seed_found and unit_id == seed_unit_id:
        return "seed"
    if seed_found:
        return "neighbor"
    return "prerequisite"


def plan_reading_order(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    *,
    seed_unit_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Plan a deterministic reading order from graph structure.

    Prerequisite-style edges are oriented so prerequisites appear before
    dependents. If a seed is provided, planning is limited to the seed's
    undirected connected component. Cycles are broken by unit id and surfaced
    in ``stats`` and item ``reason`` values.
    """
    limit_value = _validate_limit(limit)
    units_by_id = {str(unit.id): unit for unit in units}
    edge_list = list(edges)
    all_unit_ids = set(units_by_id)
    relevant_ids, seed_found = _component_for_seed(all_unit_ids, edge_list, seed_unit_id)

    order, cycle_fallback_ids = _topological_order(relevant_ids, edge_list)
    ordered_payloads = [
        _unit_payload(
            units_by_id[unit_id],
            _reason_for_unit(
                unit_id,
                seed_unit_id=seed_unit_id,
                seed_found=seed_found,
                cycle_fallback_ids=cycle_fallback_ids,
            ),
        )
        for unit_id in order
    ]

    total_planned = len(ordered_payloads)
    if limit_value is not None:
        ordered_payloads = ordered_payloads[:limit_value]

    return {
        "units": ordered_payloads,
        "stats": {
            "total_units": len(all_unit_ids),
            "planned_units": len(ordered_payloads),
            "candidate_units": total_planned,
            "omitted_units": total_planned - len(ordered_payloads),
            "seed_unit_id": seed_unit_id,
            "seed_found": seed_found,
            "cycles_detected": bool(cycle_fallback_ids),
            "cycle_fallback_count": len(cycle_fallback_ids),
            "cycle_fallback_unit_ids": sorted(cycle_fallback_ids),
            "limit": limit_value,
        },
    }
