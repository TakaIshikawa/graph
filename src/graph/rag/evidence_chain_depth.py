"""Estimate reasoning-chain depth represented by retrieved RAG evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import MISSING, content_text, iter_strings, result_id, string, tokens, value

_LINK_FIELDS = ("depends_on", "cites", "references", "parent_id", "source_id")
_RELATION_FIELDS = ("relation", "relation_label", "relation_type", "relations", "links")


def analyze_evidence_chain_depth(
    results: Iterable[Any],
    *,
    min_overlap_terms: int = 3,
) -> dict[str, Any]:
    """Return detected evidence chains, max depth, orphans, and risk warnings."""
    if isinstance(min_overlap_terms, bool) or not isinstance(min_overlap_terms, int) or min_overlap_terms < 1:
        raise ValueError("min_overlap_terms must be a positive integer")

    items = [_item(result, index) for index, result in enumerate(results or [])]
    ids = {item["id"] for item in items}
    explicit_edges, explicit_sources = _explicit_edges(items, ids)
    edges = {item["id"]: set(explicit_edges.get(item["id"], set())) for item in items}
    fallback_used = False
    for item in items:
        if item["id"] in explicit_sources:
            continue
        fallback_targets = _fallback_targets(item, items, min_overlap_terms)
        if fallback_targets:
            edges[item["id"]].update(fallback_targets)
            fallback_used = True

    cycles = _cycles(edges)
    linked_ids = {source for source, targets in edges.items() if targets} | {target for targets in edges.values() for target in targets}
    orphans = sorted((item["id"] for item in items if item["id"] not in linked_ids), key=_sort_key)
    chains = _chains(items, edges)
    max_depth = max((chain["depth"] for chain in chains), default=0)

    warnings = []
    if not items:
        warnings.append("no_results")
    if items and max_depth <= 1:
        warnings.append("shallow_evidence_chain")
    if cycles:
        warnings.append("circular_evidence_chain")
    if len(items) > 1 and not explicit_sources:
        warnings.append("missing_link_metadata")
    elif fallback_used:
        warnings.append("partial_missing_link_metadata")

    return {
        "total_results": len(items),
        "max_depth": max_depth,
        "chains": chains,
        "orphan_result_ids": orphans,
        "circular_chains": cycles,
        "edge_count": sum(len(targets) for targets in edges.values()),
        "warnings": warnings,
    }


def _item(result: Any, index: int) -> dict[str, Any]:
    return {
        "result": result,
        "index": index,
        "id": result_id(result, index),
        "tokens": tokens(content_text(result), min_length=3),
    }


def _explicit_edges(items: list[dict[str, Any]], ids: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    edges: dict[str, set[str]] = {}
    explicit_sources: set[str] = set()
    for item in items:
        targets: set[str] = set()
        for field in _LINK_FIELDS:
            for candidate in iter_strings(value(item["result"], field)):
                if candidate in ids and candidate != item["id"]:
                    targets.add(candidate)
        targets.update(_relation_targets(item["result"], ids, item["id"]))
        if targets or _has_any_relation_metadata(item["result"]):
            explicit_sources.add(item["id"])
        if targets:
            edges[item["id"]] = targets
    return edges, explicit_sources


def _relation_targets(result: Any, ids: set[str], own_id: str) -> set[str]:
    targets: set[str] = set()
    for field in _RELATION_FIELDS:
        raw = value(result, field)
        if raw is MISSING or raw is None:
            continue
        if isinstance(raw, Mapping):
            targets.update(_targets_from_mapping(raw, ids, own_id))
        elif isinstance(raw, Iterable) and not isinstance(raw, str | bytes):
            for entry in raw:
                if isinstance(entry, Mapping):
                    targets.update(_targets_from_mapping(entry, ids, own_id))
                else:
                    target = string(entry)
                    if target in ids and target != own_id:
                        targets.add(target)
        else:
            target = string(raw)
            if target in ids and target != own_id:
                targets.add(target)
    return targets


def _targets_from_mapping(raw: Mapping[str, Any], ids: set[str], own_id: str) -> set[str]:
    targets = set()
    for key in ("target", "target_id", "result_id", "source", "source_id", "parent_id", "id"):
        target = string(raw.get(key, MISSING))
        if target in ids and target != own_id:
            targets.add(target)
    return targets


def _has_any_relation_metadata(result: Any) -> bool:
    return any(value(result, field) is not MISSING for field in (*_LINK_FIELDS, *_RELATION_FIELDS))


def _fallback_targets(item: dict[str, Any], items: list[dict[str, Any]], min_overlap_terms: int) -> set[str]:
    targets = set()
    if not item["tokens"]:
        return targets
    for other in items:
        if other["id"] == item["id"] or other["index"] >= item["index"] or not other["tokens"]:
            continue
        if len(item["tokens"] & other["tokens"]) >= min_overlap_terms:
            targets.add(other["id"])
    return targets


def _chains(items: list[dict[str, Any]], edges: dict[str, set[str]]) -> list[dict[str, Any]]:
    chains = []
    for item in sorted(items, key=lambda row: _sort_key(row["id"])):
        path = _longest_path(item["id"], edges, ())
        chains.append({"root_id": item["id"], "result_ids": path, "depth": len(path)})
    chains.sort(key=lambda chain: (-chain["depth"], [_sort_key(value_) for value_ in chain["result_ids"]]))
    return chains


def _longest_path(node: str, edges: dict[str, set[str]], stack: tuple[str, ...]) -> list[str]:
    if node in stack:
        return [node]
    targets = sorted(edges.get(node, set()), key=_sort_key)
    if not targets:
        return [node]
    paths = [[node, *_longest_path(target, edges, (*stack, node))] for target in targets]
    return sorted(paths, key=lambda path: (-len(path), [_sort_key(value_) for value_ in path]))[0]


def _cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    found: set[tuple[str, ...]] = set()

    def visit(node: str, stack: tuple[str, ...]) -> None:
        if node in stack:
            cycle = stack[stack.index(node) :] + (node,)
            found.add(_canonical_cycle(cycle))
            return
        for target in sorted(edges.get(node, set()), key=_sort_key):
            visit(target, (*stack, node))

    for node in sorted(edges, key=_sort_key):
        visit(node, ())
    return [list(cycle) for cycle in sorted(found, key=lambda cycle: [_sort_key(value_) for value_ in cycle])]


def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    core = cycle[:-1]
    rotations = [core[index:] + core[:index] for index in range(len(core))]
    best = min(rotations, key=lambda item: [_sort_key(value_) for value_ in item])
    return best + (best[0],)


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is None else str(value_)
    return (text.casefold(), text)
