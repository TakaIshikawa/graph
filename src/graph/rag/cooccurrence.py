"""Local keyphrase co-occurrence maps for graph units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from graph.rag.keywords import extract_keywords
from graph.types.models import KnowledgeUnit


def _validate_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def build_keyphrase_cooccurrence(
    units: Iterable[KnowledgeUnit],
    *,
    min_count: int = 2,
    max_phrases: int = 50,
) -> dict[str, Any]:
    """Build a deterministic keyword co-occurrence map from local unit text.

    Phrases are ranked by the existing keyword extractor. Co-occurrence edge
    weights count how many units contain both selected phrases.
    """
    min_count_value = _validate_non_negative_int(min_count, "min_count")
    max_phrases_value = _validate_non_negative_int(max_phrases, "max_phrases")
    unit_list = list(units)

    keywords = extract_keywords(
        unit_list,
        min_count=min_count_value,
        limit=max_phrases_value,
    )
    phrases = [
        {
            "phrase": item["keyword"],
            "score": item["score"],
            "count": item["count"],
            "unit_count": len(item["unit_ids"]),
            "unit_ids": item["unit_ids"],
        }
        for item in keywords
    ]

    phrases_by_unit_id: dict[str, set[str]] = defaultdict(set)
    for phrase in phrases:
        phrase_text = phrase["phrase"]
        for unit_id in phrase["unit_ids"]:
            phrases_by_unit_id[unit_id].add(phrase_text)

    edge_unit_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for unit_id, unit_phrases in phrases_by_unit_id.items():
        for source, target in combinations(sorted(unit_phrases), 2):
            edge_unit_ids[(source, target)].add(unit_id)

    edges = [
        {
            "source": source,
            "target": target,
            "weight": len(unit_ids),
            "unit_ids": sorted(unit_ids),
        }
        for (source, target), unit_ids in edge_unit_ids.items()
    ]
    edges.sort(key=lambda item: (-item["weight"], item["source"], item["target"]))

    return {
        "phrases": phrases,
        "edges": edges,
        "stats": {
            "unit_count": len(unit_list),
            "phrase_count": len(phrases),
            "edge_count": len(edges),
            "min_count": min_count_value,
            "max_phrases": max_phrases_value,
        },
    }
