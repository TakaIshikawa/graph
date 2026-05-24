"""Analyze source saturation in selected RAG context packs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import domain_for, result_id, rounded_ratio, string, value


def analyze_context_source_saturation(
    context_items: Iterable[Any],
    *,
    dominance_threshold: float = 0.5,
) -> dict[str, Any]:
    """Return repetition metrics for source IDs, domains, authors, and titles."""
    if isinstance(dominance_threshold, bool) or not 0 < float(dominance_threshold) <= 1:
        raise ValueError("dominance_threshold must be between 0 and 1")
    items = list(context_items)
    total = len(items)
    dimensions = {
        "source_id": Counter(_field(item, ("source_id", "source", "source_name")) for item in items),
        "domain": Counter(domain_for(item) or "missing" for item in items),
        "author": Counter(_field(item, ("author", "authors", "creator")) for item in items),
        "title": Counter(_field(item, ("title", "source_title")) for item in items),
    }
    metrics = {name: _metric(counter, total, float(dominance_threshold)) for name, counter in dimensions.items()}
    warnings = [
        f"dominant_{name}:{metric['dominant_value']}"
        for name, metric in metrics.items()
        if metric["dominant_ratio"] > float(dominance_threshold) and metric["dominant_value"] != "missing"
    ]
    return {
        "context_count": total,
        "item_ids": [result_id(item, index) for index, item in enumerate(items)],
        "metrics": metrics,
        "dominant_sources": {name: metric for name, metric in metrics.items() if metric["dominant_ratio"] > float(dominance_threshold)},
        "warnings": warnings,
    }


def _field(item: Any, keys: tuple[str, ...]) -> str:
    for key in keys:
        text = string(value(item, key))
        if text:
            return text
    return "missing"


def _metric(counter: Counter[str], total: int, threshold: float) -> dict[str, Any]:
    if total == 0:
        return {"unique_count": 0, "dominant_value": None, "dominant_count": 0, "dominant_ratio": 0.0, "saturated": False}
    dominant_value, dominant_count = sorted(counter.items(), key=lambda item: (-item[1], item[0].casefold()))[0]
    ratio = rounded_ratio(dominant_count, total)
    return {
        "unique_count": len(counter),
        "dominant_value": dominant_value,
        "dominant_count": dominant_count,
        "dominant_ratio": ratio,
        "saturated": ratio > threshold and dominant_value != "missing",
    }
