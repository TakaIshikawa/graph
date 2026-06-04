"""Detect cookie-consent requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:cookie\s+consent|cookie\s+banner|consent|privacy|gdpr|ccpa|cmp|web\s+privacy|tracking\s+cookies?)\b",
    re.I,
)
_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("consent_banner", "high", (r"\bcookie\s+banner\b", r"\bconsent\s+banner\b", r"\bcookie\s+notice\b")),
    ("prior_consent", "high", (r"\bprior\s+consent\b", r"\bopt-?in\b", r"\bbefore\s+setting\s+cookies?\b")),
    ("cookie_categories", "medium", (r"\bcookie\s+categor(?:y|ies)\b", r"\bnecessary\s+cookies?\b", r"\banalytics\s+cookies?\b", r"\bmarketing\s+cookies?\b")),
    ("preference_center", "medium", (r"\bpreference\s+cent(?:er|re)\b", r"\bconsent\s+preferences?\b", r"\bcookie\s+settings\b")),
    ("reject_all", "high", (r"\breject\s+all\b", r"\bdecline\s+all\b", r"\brefuse\s+all\b")),
    ("consent_logging", "medium", (r"\bconsent\s+(?:log|logs|logging)\b", r"\baudit\s+logs?\b", r"\brecords?\s+of\s+consent\b")),
    ("third_party_cookies", "medium", (r"\bthird[-\s]?party\s+cookies?\b", r"\btracking\s+cookies?\b", r"\bad(?:vertising)?\s+cookies?\b")),
)


def detect_query_cookie_consent_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_cookie_consent_requirement": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_cookie_consent_requirement": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
