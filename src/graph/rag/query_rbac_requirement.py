"""Detect role-based access control requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_RBAC_CONTEXT_RE = re.compile(r"\b(?:rbac|role[-\s]?based\s+access\s+control|access\s+control|permissions?|privileges?)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("admin_roles", "high", (r"\badmin(?:istrator)?\s+roles?\b", r"\bsuper\s*admin\b", r"\bprivileged\s+roles?\b")),
    ("least_privilege", "high", (r"\bleast\s+privilege\b", r"\bminimum\s+(?:required\s+)?permissions?\b")),
    ("role_assignment", "medium", (r"\brole\s+assignments?\b", r"\bassign\s+roles?\b", r"\brole\s+membership\b")),
    ("role_permissions", "high", (r"\brole[-\s]?permission\s+matrix\b", r"\bpermissions?\s+matrix\b", r"\broles?\s+and\s+permissions?\b")),
    ("separation_of_duties", "high", (r"\bseparation\s+of\s+duties\b", r"\bsegregation\s+of\s+duties\b", r"\bconflicting\s+roles?\b")),
)


def detect_query_rbac_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _RBAC_CONTEXT_RE.search(text):
        return {"has_rbac_requirements": False, "requirements": []}
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_rbac_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
