"""Detect least privilege requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "least_privilege",
        "high",
        (
            r"\bleast[-\s]+privilege(?:d)?\b",
            r"\bprinciple\s+of\s+least\s+privilege\b",
            r"\bpolp\b",
        ),
    ),
    (
        "minimum_permissions",
        "high",
        (
            r"\bminimum\s+(?:required\s+|necessary\s+)?permissions?\b",
            r"\bminimal\s+(?:required\s+|necessary\s+)?permissions?\b",
            r"\bonly\s+(?:the\s+)?permissions?\s+(?:required|needed|necessary)\b",
        ),
    ),
    (
        "need_to_know_access",
        "medium",
        (
            r"\bneed[-\s]+to[-\s]+know\s+access\b",
            r"\baccess\s+on\s+a\s+need[-\s]+to[-\s]+know\s+basis\b",
            r"\blimit\s+access\s+to\s+(?:users?|people|staff|personnel)\s+who\s+need\s+to\s+know\b",
        ),
    ),
    (
        "privilege_reduction",
        "medium",
        (
            r"\breduce\s+(?:user\s+|account\s+|role\s+)?privileges?\b",
            r"\blimit\s+(?:user\s+|account\s+|role\s+)?privileges?\b",
            r"\bremove\s+excessive\s+(?:permissions?|privileges?)\b",
        ),
    ),
    (
        "excessive_access_review",
        "medium",
        (
            r"\breview\s+(?:for\s+)?excessive\s+access\b",
            r"\bdetect\s+excessive\s+(?:access|permissions?|privileges?)\b",
            r"\bidentify\s+over[-\s]?privileged\s+(?:users?|accounts?|roles?)\b",
        ),
    ),
)


def detect_query_least_privilege_requirement(query: str) -> dict[str, Any]:
    """Return least privilege requirement signals mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append(
                {
                    "category": category,
                    "matched_text": match.group(0),
                    "severity": severity,
                    "evidence_terms": _evidence_terms(match.group(0)),
                }
            )
    requirements.sort(key=lambda row: row["category"])
    return {
        "requires_least_privilege": bool(requirements),
        "classification": "least_privilege_requirement" if requirements else "unrelated",
        "requirements": requirements,
        "evidence_terms": sorted({term for row in requirements for term in row["evidence_terms"]}),
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _evidence_terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
