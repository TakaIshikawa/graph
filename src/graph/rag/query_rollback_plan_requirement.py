"""Detect rollback plan requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ROLLBACK_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rollback", (r"\brollback\b", r"\broll\s+back\b")),
    ("backout_plan", (r"\bbackout\s+plan\b", r"\bback[-\s]out\s+procedure\b")),
    ("revert", (r"\brevert\s+(?:the\s+)?(?:deployment|release|change)\b",)),
    ("restore_previous_version", (r"\brestore\s+(?:the\s+)?previous\s+version\b",)),
    ("feature_flag_disablement", (r"\bdisable\s+(?:the\s+)?feature\s+flag\b", r"\bturn\s+off\s+(?:the\s+)?feature\s+flag\b")),
    ("canary_abort", (r"\babort\s+(?:the\s+)?canary\b", r"\bcanary\s+abort\b")),
    ("migration_rollback", (r"\bmigration\s+rollback\b", r"\broll\s+back\s+(?:the\s+)?migration\b")),
    ("recovery_checkpoint", (r"\brecovery\s+checkpoint\b", r"\brestore\s+checkpoint\b")),
)
_RISK_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("deployment", (r"\bdeployment\b", r"\brelease\b")),
    ("migration", (r"\bmigration\b",)),
    ("feature_flag", (r"\bfeature\s+flag\b",)),
)


def detect_query_rollback_plan_requirement(query: str) -> dict[str, Any]:
    """Return rollback plan requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    rollback_terms = [term for term, patterns in _ROLLBACK_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    if "migration_rollback" in rollback_terms and "rollback" in rollback_terms:
        rollback_terms.remove("rollback")
    risk_terms = [term for term, patterns in _RISK_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    recommendations = ["document rollback steps"] if rollback_terms else []
    if "migration" in risk_terms:
        recommendations.append("verify migration recovery point")
    return {
        "requires_rollback_plan": bool(rollback_terms),
        "rollback_terms": rollback_terms,
        "matched_phrases": rollback_terms,
        "risk_terms": risk_terms,
        "recommendations": recommendations,
        "confidence": "high" if any(term in {"rollback", "backout_plan", "migration_rollback"} for term in rollback_terms) else "medium" if rollback_terms else "none",
    }
