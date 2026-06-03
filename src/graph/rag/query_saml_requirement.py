"""Detect SAML SSO requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_PATTERNS = (
    r"\bsaml\b",
    r"\bsso\b",
    r"\bsingle\s+sign[-\s]?on\b",
    r"\bidentity\b",
    r"\bauthentication\b",
    r"\bidentity\s+provider\b",
    r"\bidp\b",
    r"\bservice\s+provider\b",
    r"\bsp\b",
)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("assertion_signing", "high", (r"\bsigned?\s+saml\s+assertions?\b", r"\bassertion\s+signing\b", r"\bsign\s+(?:the\s+)?assertions?\b")),
    ("attribute_mapping", "medium", (r"\battribute\s+mapping\b", r"\bmap\s+(?:saml\s+)?attributes?\b", r"\bgroup\s+attribute\b", r"\bemail\s+attribute\b")),
    ("certificate_rotation", "medium", (r"\bcertificate\s+rotation\b", r"\brotate\s+(?:saml\s+)?certificates?\b", r"\bcertificate\s+rollover\b")),
    ("idp_initiated_login", "medium", (r"\bidp[-\s]?initiated\b", r"\bidentity\s+provider[-\s]?initiated\b")),
    ("metadata_xml", "high", (r"\bmetadata\s+xml\b", r"\bsaml\s+metadata\b", r"\bidp\s+metadata\b", r"\bsp\s+metadata\b")),
    ("name_id_format", "medium", (r"\bname\s*id\s+format\b", r"\bnameid\s+format\b", r"\bname\s*identifier\s+format\b")),
    ("sp_initiated_login", "medium", (r"\bsp[-\s]?initiated\b", r"\bservice\s+provider[-\s]?initiated\b")),
)


def detect_query_saml_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _has_identity_context(text):
        return {"has_saml_requirements": False, "requirements": []}
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_saml_requirements": bool(requirements), "requirements": requirements}


def _has_identity_context(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in _CONTEXT_PATTERNS)


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
