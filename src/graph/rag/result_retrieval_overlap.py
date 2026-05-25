"""Analyze overlap between multiple retrieval strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from graph.rag._analysis_utils import result_id


def analyze_retrieval_overlap(result_sets: Mapping[str, list[Any] | tuple[Any, ...] | set[Any]]) -> dict[str, Any]:
    """Compute unique, shared, pairwise, and consensus result ids by strategy."""
    strategies = list((result_sets or {}).keys())
    ids_by_strategy = {name: _ids(result_sets.get(name, [])) for name in strategies}
    all_ids = sorted(set().union(*ids_by_strategy.values()) if ids_by_strategy else [])
    consensus = sorted((set.intersection(*ids_by_strategy.values()) if ids_by_strategy else set()))
    pairwise = []
    for index, left in enumerate(strategies):
        for right in strategies[index + 1 :]:
            left_ids = ids_by_strategy[left]
            right_ids = ids_by_strategy[right]
            union = left_ids | right_ids
            overlap = left_ids & right_ids
            pairwise.append(
                {
                    "left_strategy": left,
                    "right_strategy": right,
                    "overlap_ids": sorted(overlap),
                    "overlap_count": len(overlap),
                    "overlap_ratio": 0.0 if not union else round(len(overlap) / len(union), 4),
                }
            )
    return {
        "strategy_ids": {name: sorted(ids) for name, ids in ids_by_strategy.items()},
        "unique_ids_by_strategy": {name: sorted(ids - set().union(*(other for key, other in ids_by_strategy.items() if key != name))) for name, ids in ids_by_strategy.items()},
        "overlapping_ids": [item for item in all_ids if sum(item in ids for ids in ids_by_strategy.values()) > 1],
        "consensus_ids": consensus,
        "pairwise_overlap": pairwise,
    }


def _ids(results: Any) -> set[str]:
    ids = set()
    for index, item in enumerate(results or []):
        ids.add(item if isinstance(item, str) else result_id(item, index))
    return ids
