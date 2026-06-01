"""Detect maintenance window requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_TERM_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("maintenance_window", (r"\bmaintenance\s+window\b",)),
    ("scheduled_downtime", (r"\bscheduled\s+downtime\b", r"\bplanned\s+outage\b")),
    ("blackout_period", (r"\bblackout\s+period\b", r"\bchange\s+blackout\b")),
    ("freeze_window", (r"\bfreeze\s+window\b", r"\bdeployment\s+freeze\b")),
    ("after_hours", (r"\bafter[-\s]hours\s+change\b", r"\boff[-\s]hours\s+maintenance\b")),
    ("weekend_deployment", (r"\bweekend\s+deployment\b", r"\bdeploy\s+on\s+the\s+weekend\b")),
    ("no_downtime", (r"\bno[-\s]downtime\s+maintenance\b", r"\bzero[-\s]downtime\b")),
    ("customer_notice", (r"\bcustomer\s+notice\b", r"\bnotify\s+customers\b")),
)
_TIME_PATTERN = re.compile(r"\b(?:\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\s*(?:-|to)\s*(?:\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b", re.I)


def detect_query_maintenance_window_requirement(query: str) -> dict[str, Any]:
    """Return maintenance-window requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    window_terms = [term for term, patterns in _TERM_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    time_windows = [match.group(0) for match in _TIME_PATTERN.finditer(text)]
    recommendations = []
    if window_terms:
        recommendations.append("schedule maintenance window")
    if "customer_notice" in window_terms:
        recommendations.append("prepare customer notice")
    return {
        "requires_maintenance_window": bool(window_terms),
        "window_terms": window_terms,
        "time_windows": time_windows,
        "matched_phrases": window_terms,
        "recommendations": recommendations,
        "confidence": "high" if any(term in {"maintenance_window", "scheduled_downtime"} for term in window_terms) else "medium" if window_terms else "none",
    }
