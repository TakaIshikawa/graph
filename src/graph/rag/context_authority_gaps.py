"""Analyze authority tiers missing from retrieved RAG context."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, domain_for, result_id, string, value

_HIGH_STAKES_RE = re.compile(r"\b(medical|diagnosis|dose|legal|lawsuit|contract|financial|investment|tax|loan)\b", re.I)
_ORDERED_TIERS = ("primary", "expert", "official", "publisher", "community", "unknown")


def analyze_context_authority_gaps(results: Iterable[Any], *, required_tiers: Iterable[str] | None = None, query: str | None = None) -> dict[str, Any]:
    """Return present and missing authority tiers, with explicit metadata first."""
    rows = []
    for index, result in enumerate(results):
        tier, reason = _tier(result)
        rows.append({"result_id": result_id(result, index), "authority_tier": tier, "reason": reason})

    required = _required(required_tiers, query)
    present = sorted({row["authority_tier"] for row in rows if row["authority_tier"] != "unknown"}, key=_tier_key)
    missing = [tier for tier in required if tier not in present]
    counts = Counter(row["authority_tier"] for row in rows)
    tier_counts = {tier: counts.get(tier, 0) for tier in _ORDERED_TIERS}
    warnings = []
    if not rows:
        warnings.append("no_results")
    if missing:
        warnings.append("missing_required_authority_tiers")
    if tier_counts["unknown"]:
        warnings.append("unknown_authority_tier")
    return {
        "total_results": len(rows),
        "required_tiers": required,
        "present_tiers": present,
        "missing_tiers": missing,
        "tier_counts": tier_counts,
        "results": rows,
        "warnings": warnings,
    }


def _required(required_tiers: Iterable[str] | None, query: str | None) -> list[str]:
    if required_tiers is not None:
        return _normalize_tiers(required_tiers)
    if query and _HIGH_STAKES_RE.search(str(query)):
        return ["primary", "expert"]
    return ["primary"]


def _tier(result: Any) -> tuple[str, str]:
    explicit = _text(_first_value(result, ("authority_tier", "source_type")))
    aliases = {
        "primary": "primary",
        "official": "primary",
        "government": "primary",
        "regulator": "primary",
        "expert": "expert",
        "academic": "expert",
        "journal": "expert",
        "peer_reviewed": "expert",
        "publisher": "publisher",
        "news": "publisher",
        "documentation": "official",
        "docs": "official",
        "community": "community",
        "forum": "community",
        "blog": "community",
        "social": "community",
    }
    if explicit in aliases:
        return aliases[explicit], "explicit_metadata"
    domain = domain_for(result)
    if domain and domain.endswith(".gov"):
        return "primary", "domain_heuristic"
    if domain and domain.endswith(".edu"):
        return "expert", "domain_heuristic"
    if domain:
        return "publisher", "domain_present"
    return "unknown", "missing_authority_metadata"


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        item = value(result, key)
        if item is not MISSING and item is not None and string(item) is not None:
            return item
    return MISSING


def _text(value_: Any) -> str | None:
    text = string(value_)
    return None if text is None else text.casefold().replace("-", "_").replace(" ", "_")


def _normalize_tiers(tiers: Iterable[str]) -> list[str]:
    seen = set()
    normalized = []
    for tier in tiers:
        text = _text(tier)
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def _tier_key(tier: str) -> int:
    return _ORDERED_TIERS.index(tier) if tier in _ORDERED_TIERS else len(_ORDERED_TIERS)
