"""Detect cookie security requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_COOKIE_RE = re.compile(r"\bcookies?\b", re.I)
_REQUIREMENT_RE = re.compile(r"\b(?:require|set|configure|must|should|security|protect|signed|encrypted|httponly|samesite|secure)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("httponly", "high", (r"\bhttp\s*only\b", r"\bhttponly\b")),
    ("secure", "high", (r"\bsecure\s+flag\b", r"\bsecure\s+cookies?\b", r"\bcookies?\s+secure\b")),
    ("samesite", "high", (r"\bsame\s*site\b", r"\bsamesite\b")),
    ("scope", "medium", (r"\bdomain\s+(?:scope|attribute)\b", r"\bpath\s+(?:scope|attribute)\b", r"\bcookie\s+(?:domain|path)\b")),
    ("expiration", "medium", (r"\bmax[-\s]?age\b", r"\bexpires?\b", r"\bexpiration\b", r"\bcookie\s+ttl\b")),
    ("signed_encrypted", "high", (r"\bsigned\s+cookies?\b", r"\bencrypted\s+cookies?\b", r"\bcookie\s+signing\b", r"\bcookie\s+encryption\b")),
)


def detect_query_cookie_security_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_cookie_requirement = bool(_COOKIE_RE.search(text) and _REQUIREMENT_RE.search(text))
    requirements = []
    if has_cookie_requirement:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_cookie_security_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
