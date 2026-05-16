"""Score repeated or near-duplicate evidence snippets in RAG results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _group_components(matches: list[tuple[int, int, float]], total: int) -> list[list[int]]:
    parent = list(range(total))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right, _score in matches:
        union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(total):
        groups.setdefault(find(index), []).append(index)
    return [group for group in groups.values() if len(group) > 1]


def score_evidence_redundancy(
    results: Iterable[Any],
    *,
    overlap_threshold: float = 0.65,
) -> dict[str, Any]:
    """Return duplicate evidence groups and an overall redundancy score."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    try:
        threshold = float(overlap_threshold)
    except (TypeError, ValueError):
        threshold = 0.65
    threshold = min(max(threshold, 0.0), 1.0)

    normalized = []
    for index, result in enumerate(rows):
        term_set = tokens(content_text(result))
        normalized.append(
            {
                "index": index,
                "result_id": result_id(result, index),
                "tokens": term_set if len(term_set) >= 3 else set(),
            }
        )

    matches: list[tuple[int, int, float]] = []
    compared_count = 0
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            if not normalized[left]["tokens"] or not normalized[right]["tokens"]:
                continue
            compared_count += 1
            score = _jaccard(normalized[left]["tokens"], normalized[right]["tokens"])
            if score >= threshold:
                matches.append((left, right, round(score, 4)))

    duplicate_groups = []
    duplicate_indexes: set[int] = set()
    for group in _group_components(matches, len(normalized)):
        duplicate_indexes.update(group)
        group_matches = [
            score
            for left, right, score in matches
            if left in group and right in group
        ]
        duplicate_groups.append(
            {
                "result_ids": [normalized[index]["result_id"] for index in group],
                "max_overlap": max(group_matches) if group_matches else 0.0,
                "pair_count": len(group_matches),
            }
        )

    distinct_count = len(normalized) - len(duplicate_indexes) + len(duplicate_groups)
    redundancy_score = 0.0
    if normalized:
        redundancy_score = round((len(normalized) - distinct_count) / len(normalized), 4)

    return {
        "duplicate_groups": duplicate_groups,
        "redundancy_score": redundancy_score,
        "compared_count": compared_count,
        "distinct_count": distinct_count,
    }
