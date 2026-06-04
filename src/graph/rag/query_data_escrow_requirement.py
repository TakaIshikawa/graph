"""Detect data and source-code escrow requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(r"\b(?:escrow|source[-\s]?code\s+escrow|data\s+escrow)\b", re.I)

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("beneficiary_access", "high", (r"\bbeneficiary\s+access\b", r"\bbeneficiar(?:y|ies)\b", r"\baccess\s+to\s+escrow\b")),
    ("deposit_scope", "high", (r"\bescrow\s+deposits?\b", r"\bdeposit\s+scope\b", r"\bsource[-\s]?code\s+escrow\b", r"\bdata\s+escrow\b")),
    ("escrow_agent", "high", (r"\bthird[-\s]?party\s+escrow\s+agent\b", r"\bescrow\s+agent\b")),
    ("release_conditions", "high", (r"\brelease\s+conditions?\b", r"\brelease\s+triggers?\b", r"\bconditions?\s+for\s+release\b")),
    ("verification", "medium", (r"\bverification\s+of\s+deposits?\b", r"\bverify\s+deposits?\b", r"\bdeposit\s+verification\b")),
)


def detect_query_data_escrow_requirements(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_data_escrow_requirements": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_data_escrow_requirements": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
