"""Detect certificate lifecycle management requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("renewal", (r"\bcertificate\s+renewal\b", r"\brenew\s+certificates?\b")),
    ("expiration_monitoring", (r"\bcertificate\s+expir(?:y|ation)\s+monitoring\b", r"\bmonitor\s+certificate\s+expir(?:y|ation)\b")),
    ("automation", (r"\bacme\b", r"\bautomated\s+certificate\b", r"\bcertificate\s+automation\b")),
    ("ca_rotation", (r"\bca\s+rotation\b", r"\bcertificate\s+authority\s+rotation\b")),
    ("mtls_lifecycle", (r"\bmtls\s+certificate\s+lifecycle\b", r"\bmutual\s+tls\s+certificate\s+lifecycle\b")),
)


def detect_query_certificate_lifecycle_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {"requires_certificate_lifecycle": bool(categories), "cue_categories": categories}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
