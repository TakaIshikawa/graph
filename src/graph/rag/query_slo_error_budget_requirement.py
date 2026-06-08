"""Detect SLO error budget requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("slo", (r"\bSLOs?\b", r"\bservice\s+level\s+objectives?\b")),
    ("error_budget", (r"\berror\s+budgets?\b",)),
    ("burn_rate", (r"\bburn\s+rates?\b", r"\berror\s+budget\s+burn\b")),
    ("reliability_objective", (r"\breliability\s+objectives?\b", r"\breliability\s+target\b")),
    ("sli", (r"\bSLIs?\b", r"\bservice\s+level\s+indicators?\b")),
    ("alert_threshold", (r"\balert\s+thresholds?\b", r"\balerting\s+thresholds?\b")),
)


def detect_query_slo_error_budget_requirement(query: str) -> dict[str, Any]:
    """Return SLO error-budget requirement categories mentioned by a query."""
    requirements = _requirements(query)
    return {"has_slo_error_budget_requirement": bool(requirements), "requirements": requirements}


def _requirements(query: str) -> list[dict[str, str]]:
    text = " ".join(str(query or "").split())
    rows: list[dict[str, str]] = []
    for category, patterns in _CATEGORIES:
        match = _first_match(patterns, text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0)})
    return sorted(rows, key=lambda row: row["category"])


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None
