"""Plan a source-balanced subset of RAG context results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, number, result_id, rounded_ratio, string, value


def plan_context_source_limits(
    results: Iterable[Any],
    *,
    max_results: int = 12,
    max_per_source: int = 3,
) -> dict[str, Any]:
    """Keep high-scoring results while capping per-source representation."""
    for name, setting in (("max_results", max_results), ("max_per_source", max_per_source)):
        if not isinstance(setting, int) or isinstance(setting, bool) or setting < 1:
            raise ValueError(f"{name} must be a positive integer")

    candidates = [_candidate(result, index) for index, result in enumerate(results)]
    ordered = sorted(candidates, key=lambda item: (-(item["score"] if item["score"] is not None else float("-inf")), item["rank"], _sort_key(item["id"])))
    kept = []
    deferred = []
    counts: Counter[str] = Counter()
    for item in ordered:
        if len(kept) >= max_results:
            deferred.append((item, "over_max_results"))
        elif counts[item["source"]] >= max_per_source:
            deferred.append((item, "over_source_limit"))
        else:
            kept.append(item)
            counts[item["source"]] += 1

    source_totals = Counter(item["source"] for item in candidates)
    warnings = []
    total = len(candidates)
    if any(rounded_ratio(count, total) >= 0.5 and total > 1 for count in source_totals.values()):
        warnings.append("heavy_source_concentration")

    return {
        "total_results": total,
        "kept_ids": [item["id"] for item in sorted(kept, key=lambda item: item["rank"])],
        "deferred_ids": [item["id"] for item, _ in sorted(deferred, key=lambda pair: pair[0]["rank"])],
        "deferred": [
            {"id": item["id"], "source": item["source"], "reason": reason}
            for item, reason in sorted(deferred, key=lambda pair: pair[0]["rank"])
        ],
        "source_counts": dict(sorted(counts.items())),
        "source_totals": dict(sorted(source_totals.items())),
        "warnings": warnings,
    }


def _candidate(result: Any, index: int) -> dict[str, Any]:
    return {
        "id": result_id(result, index),
        "source": string(_first_value(result, ("source_project", "source", "source_id"))) or "unknown",
        "score": number(value(result, "score")),
        "rank": index,
    }


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        item = value(result, key)
        if item is not MISSING and item is not None and string(item) is not None:
            return item
    return MISSING


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is None else str(value_)
    return (text.casefold(), text)
