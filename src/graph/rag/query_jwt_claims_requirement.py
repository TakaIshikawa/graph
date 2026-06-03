"""Detect JWT claim requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_JWT_CONTEXT_RE = re.compile(r"\b(?:jwt|json\s+web\s+token|access\s+token|id\s+token|bearer\s+token|token\s+claims?)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("audience", "high", (r"\baudience\b", r"\baud\b")),
    ("custom_claims", "medium", (r"\bcustom\s+claims?\b", r"\bapplication[-\s]?specific\s+claims?\b")),
    ("expiration", "high", (r"\bexpiration\b", r"\bexpires?\b", r"\bexp\b", r"\bttl\b")),
    ("issuer", "high", (r"\bissuer\b", r"\biss\b")),
    ("scopes_roles", "medium", (r"\bscopes?\b", r"\broles?\s+claims?\b", r"\bgroups?\s+claims?\b")),
    ("subject", "high", (r"\bsubject\b", r"\bsub\b")),
)


def detect_query_jwt_claims_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _JWT_CONTEXT_RE.search(text):
        return {"has_jwt_claims_requirements": False, "requirements": []}
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_jwt_claims_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
