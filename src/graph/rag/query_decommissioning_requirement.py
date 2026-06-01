"""Detect decommissioning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_DECOMMISSIONING_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("decommission", (r"\bdecommission(?:ing)?\b",)),
    ("end_of_life", (r"\bend[-\s]?of[-\s]?life\b", r"\beol\b")),
    ("retire_system", (r"\bretire\s+(?:the\s+)?(?:system|service|application|app|platform)\b", r"\bsystem\s+retirement\b")),
    ("sunset", (r"\bsunset\s+(?:the\s+)?(?:system|service|application|app|platform)\b", r"\bsystem\s+sunset\b")),
)
_LIFECYCLE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("archive", (r"\barchive\s+(?:before\s+)?(?:removal|shutdown|decommissioning)?\b", r"\bdata\s+archive\b")),
    ("migrate", (r"\bmigrat(?:e|ion)\s+off\b", r"\bmigrate\s+(?:users|data|traffic)\b")),
    ("replace", (r"\breplacement\s+rollout\b", r"\breplace\s+(?:the\s+)?(?:system|service|application|app|platform)\b")),
    ("shutdown", (r"\bshutdown\s+plan\b", r"\bshutdown\s+(?:the\s+)?(?:system|service|application|app|platform)\b")),
)


def detect_query_decommissioning_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    decommissioning_terms, matched_phrases = _collect_matches(_DECOMMISSIONING_TERMS, text)
    lifecycle_terms, lifecycle_matches = _collect_matches(_LIFECYCLE_TERMS, text)
    matched_phrases.extend(lifecycle_matches)
    requires_plan = bool(decommissioning_terms)
    return {
        "requires_decommissioning_plan": requires_plan,
        "decommissioning_terms": decommissioning_terms,
        "matched_phrases": matched_phrases,
        "lifecycle_terms": lifecycle_terms,
        "recommendations": _recommendations(requires_plan, lifecycle_terms),
        "confidence": "high" if requires_plan and lifecycle_terms else ("medium" if requires_plan else "none"),
    }


def _collect_matches(specs: tuple[tuple[str, tuple[str, ...]], ...], text: str) -> tuple[list[str], list[str]]:
    terms = []
    phrases = []
    for term, patterns in specs:
        match = _first_match(patterns, text)
        if match:
            terms.append(term)
            phrases.append(match.group(0))
    return terms, phrases


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _recommendations(requires_plan: bool, lifecycle_terms: list[str]) -> list[str]:
    if not requires_plan:
        return []
    recommendations = ["define_owner_and_timeline", "confirm_user_and_data_migration"]
    if "archive" in lifecycle_terms:
        recommendations.append("verify_archive_before_removal")
    if "replace" in lifecycle_terms:
        recommendations.append("coordinate_replacement_rollout")
    return recommendations


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
