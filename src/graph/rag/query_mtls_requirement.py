"""Detect mutual TLS requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_MTLS_CONTEXT_RE = re.compile(r"\b(?:mtls|m\s*tls|mutual\s+tls|mutual\s+transport\s+layer\s+security|client\s+certificates?)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ca_trust", "high", (r"\bca\s+trust\b", r"\bcertificate\s+authorit(?:y|ies)\b", r"\btrusted\s+cas?\b", r"\btrust\s+store\b")),
    ("client_certificate", "high", (r"\bclient\s+certificates?\b", r"\bclient\s+certs?\b", r"\bpresent\s+(?:a\s+)?certificate\b")),
    ("handshake_validation", "high", (r"\btls\s+handshake\b", r"\bhandshake\s+validation\b", r"\bvalidate\s+(?:the\s+)?handshake\b")),
    ("rotation", "medium", (r"\bcertificate\s+rotation\b", r"\brotate\s+certificates?\b", r"\bcertificate\s+renewal\b")),
    ("service_authentication", "high", (r"\bservice[-\s]?to[-\s]?service\s+authentication\b", r"\bmutual\s+authentication\b", r"\bworkload\s+authentication\b")),
)


def detect_query_mtls_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _MTLS_CONTEXT_RE.search(text):
        return {"has_mtls_requirements": False, "requirements": []}
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_mtls_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
