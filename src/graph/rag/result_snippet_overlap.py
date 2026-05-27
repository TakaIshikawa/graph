"""Measure lexical overlap between retrieved result snippets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, rounded_ratio, tokens


def audit_result_snippet_overlap(results: Iterable[Any], *, threshold: float = 0.5) -> dict[str, Any]:
    """Return redundant result pairs whose normalized token overlap exceeds threshold."""
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    items = list(results or [])
    token_sets = [tokens(content_text(item)) for item in items]
    pairs = []
    ratios = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            union = token_sets[left] | token_sets[right]
            ratio = 0.0 if not union else round(len(token_sets[left] & token_sets[right]) / len(union), 4)
            ratios.append(ratio)
            if ratio >= threshold:
                pairs.append(
                    {
                        "left_id": result_id(items[left], left),
                        "right_id": result_id(items[right], right),
                        "overlap_ratio": ratio,
                    }
                )
    pairs.sort(key=lambda pair: (-pair["overlap_ratio"], pair["left_id"], pair["right_id"]))

    return {
        "result_count": len(items),
        "overlapping_pair_count": len(pairs),
        "max_overlap_ratio": max(ratios) if ratios else 0.0,
        "average_overlap_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
        "overlapping_pairs": pairs,
    }
