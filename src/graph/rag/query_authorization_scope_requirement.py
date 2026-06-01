"""Detect authorization scope requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("abac", re.compile(r"\b(?:abac|attribute[-\s]?based\s+access\s+control|attribute\s+policy)\b", re.I), "high"),
    ("admin_only", re.compile(r"\b(?:admin[-\s]?only|administrators?\s+only|only\s+admins?)\b", re.I), "high"),
    ("least_privilege", re.compile(r"\b(?:least\s+privilege|minimum\s+(?:necessary\s+)?permissions?|need[-\s]?to[-\s]?know)\b", re.I), "high"),
    ("permission_boundary", re.compile(r"\b(?:permission\s+boundar(?:y|ies)|bounded\s+permissions?|scope\s+boundar(?:y|ies))\b", re.I), "high"),
    ("rbac", re.compile(r"\b(?:rbac|role[-\s]?based\s+access\s+control)\b", re.I), "high"),
    ("role_permissions", re.compile(r"\b(?:role\s+permissions?|permissions?\s+by\s+role|roles?\s+can\s+(?:read|write|delete|access|approve))\b", re.I), "medium"),
    ("row_level_security", re.compile(r"\b(?:row[-\s]?level\s+security|rls|tenant[-\s]?scoped\s+rows?)\b", re.I), "high"),
    ("scoped_tokens", re.compile(r"\b(?:(?:scoped|limited[-\s]?scope)\s+tokens?|token\s+scopes?|oauth\s+scopes?)\b", re.I), "medium"),
)


def detect_query_authorization_scope_requirements(query: str) -> list[dict[str, Any]]:
    """Return authorization scope requirement matches mentioned by a query."""
    normalized = _normalize_query(query)
    rows = []
    for category, pattern, severity in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "severity": severity})
    rows.sort(key=lambda row: row["category"])
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
