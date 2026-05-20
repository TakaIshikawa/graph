"""Summarize authority tiers represented by RAG retrieval results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, domain_for, result_id, rounded_ratio, string, value

_TIER_SCORES = {"high": 1.0, "medium": 0.6, "low": 0.25, "unknown": 0.0}


def analyze_source_authority_mix(results: Iterable[Any]) -> dict[str, Any]:
    """Infer authority tiers from explicit metadata before domain heuristics."""
    rows = []
    for index, result in enumerate(results):
        tier, reason = _tier(result)
        rows.append({"result_id": result_id(result, index), "tier": tier, "reason": reason})

    total = len(rows)
    counts = Counter(row["tier"] for row in rows)
    tier_counts = {tier: counts.get(tier, 0) for tier in ("high", "medium", "low", "unknown")}
    tier_ratios = {tier: rounded_ratio(count, total) for tier, count in tier_counts.items()}
    dominant_tier = None if total == 0 else sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    score = round(sum(_TIER_SCORES[row["tier"]] for row in rows) / total, 4) if total else 0.0

    warnings = []
    if total == 0:
        warnings.append("no_results")
    if counts["unknown"]:
        warnings.append("anonymous_or_missing_authority")
    if total and counts["low"] / total >= 0.25:
        warnings.append("low_authority_mix")
    if total > 1 and dominant_tier and counts[dominant_tier] / total >= 0.75:
        warnings.append("over_concentrated_authority_tier")

    return {
        "total_results": total,
        "tier_counts": tier_counts,
        "tier_ratios": tier_ratios,
        "dominant_tier": dominant_tier,
        "authority_score": score,
        "results": sorted(rows, key=lambda row: _sort_key(row["result_id"])),
        "warnings": warnings,
    }


def _tier(result: Any) -> tuple[str, str]:
    source_type = _text(_first_value(result, ("source_type", "type")))
    if _truthy(value(result, "peer_reviewed")) or source_type in {"peer_reviewed", "academic", "journal", "government", "official"}:
        return "high", "explicit_high_authority"
    if _truthy(value(result, "verified")) or source_type in {"documentation", "docs", "news", "publisher", "book"}:
        return "medium", "explicit_verified_or_publisher"
    if source_type in {"forum", "social", "blog", "comment", "user_generated"}:
        return "low", "explicit_low_authority"

    domain = domain_for(result)
    if domain and (domain.endswith(".gov") or domain.endswith(".edu")):
        return "high", "domain_heuristic"
    if domain:
        return "medium", "domain_present"
    if string(value(result, "author")) or string(value(result, "publisher")):
        return "medium", "attributed_author"
    return "unknown", "missing_authority_metadata"


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        item = value(result, key)
        if item is not MISSING and item is not None and string(item) is not None:
            return item
    return MISSING


def _truthy(value_: Any) -> bool:
    if isinstance(value_, bool):
        return value_
    return _text(value_) in {"1", "true", "yes", "y", "verified", "peer_reviewed"}


def _text(value_: Any) -> str | None:
    text = string(value_)
    return None if text is None else text.casefold().replace("-", "_").replace(" ", "_")


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is None else str(value_)
    return (text.casefold(), text)
