"""Detect breach-notification requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:data\s+breach|security\s+breach|breach\s+notification|security\s+incident|privacy\s+incident|incident\s+response)\b",
    re.I,
)
_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("notification_deadline", "high", (r"\b(?:within|in|by)\s+\d+\s+(?:hours?|days?)\b", r"\bnotification\s+deadlines?\b")),
    ("regulator_notice", "high", (r"\bregulator(?:y)?\s+notice\b", r"\bnotify\s+regulators?\b", r"\bsupervisory\s+authorit(?:y|ies)\b")),
    ("customer_notice", "high", (r"\bcustomer\s+notice\b", r"\bnotify\s+customers?\b", r"\baffected\s+(?:users?|customers?)\b")),
    ("affected_data_scope", "high", (r"\baffected\s+data\b", r"\bdata\s+scope\b", r"\bpersonal\s+data\s+involved\b")),
    ("law_enforcement_delay", "medium", (r"\blaw\s+enforcement\s+delay\b", r"\bdelay\s+notice\s+for\s+law\s+enforcement\b")),
    ("communication_template", "medium", (r"\bcommunication\s+templates?\b", r"\bnotice\s+templates?\b", r"\bnotification\s+templates?\b")),
    ("incident_severity_threshold", "high", (r"\bseverity\s+thresholds?\b", r"\bmaterial\s+breach\b", r"\brisk\s+thresholds?\b")),
)


def detect_query_breach_notification_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_breach_notification_requirement": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_breach_notification_requirement": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
