"""Detect persona and personal-context cues in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ROLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("founder", re.compile(r"\bas a founder\b|\bfounder\b", re.I)),
    ("student", re.compile(r"\bas a student\b|\bstudent\b", re.I)),
    ("team", re.compile(r"\bfor my team\b|\bour team\b", re.I)),
    ("company", re.compile(r"\bour company\b|\bmy company\b", re.I)),
    ("doctor", re.compile(r"\bmy doctor\b", re.I)),
    ("family", re.compile(r"\bmy family\b|\bour family\b", re.I)),
)
_OWNERSHIP: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("my", re.compile(r"\bmy\s+\w+", re.I)),
    ("our", re.compile(r"\bour\s+\w+", re.I)),
    ("internal_policy", re.compile(r"\binternal policy\b|\bcompany policy\b", re.I)),
    ("portfolio", re.compile(r"\bmy portfolio\b", re.I)),
)
_PRIVACY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("health", re.compile(r"\bmy doctor\b|\bmedical\b|\bhealth\b", re.I)),
    ("family", re.compile(r"\bmy family\b|\bour family\b", re.I)),
    ("financial", re.compile(r"\bmy portfolio\b|\bmy taxes\b|\bbank account\b", re.I)),
    ("internal", re.compile(r"\binternal policy\b|\bour company\b|\bmy company\b", re.I)),
)


def detect_query_persona_context(query: str) -> dict[str, Any]:
    """Return persona roles, ownership cues, and privacy-sensitive context."""
    normalized = _normalize_query(query)
    roles = [label for label, pattern in _ROLES if pattern.search(normalized)]
    ownership = [label for label, pattern in _OWNERSHIP if pattern.search(normalized)]
    privacy = [label for label, pattern in _PRIVACY if pattern.search(normalized)]
    has_context = bool(roles or ownership or privacy)
    recommendations = []
    if roles:
        recommendations.append("adapt_retrieval_and_answer_framing_to_declared_role")
    if ownership:
        recommendations.append("preserve_user_specific_context_without_overgeneralizing")
    if privacy:
        recommendations.append("avoid_exposing_private_context_in_query_logs_or_citations")
    return {
        "has_persona_context": has_context,
        "persona_roles": roles,
        "ownership_cues": ownership,
        "privacy_cues": privacy,
        "recommendations": recommendations,
        "confidence": 0.85 if roles or privacy else (0.55 if ownership else 0.0),
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())
