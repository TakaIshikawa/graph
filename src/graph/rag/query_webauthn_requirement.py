"""Detect WebAuthn and FIDO2 requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_WEBAUTHN_CONTEXT_RE = re.compile(r"\b(?:webauthn|web\s+authn|fido2?|passkeys?)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("attestation", "high", (r"\battestation\b", r"\bauthenticator\s+attestation\b")),
    ("challenge", "high", (r"\bchallenge\s+(?:handling|validation|response)\b", r"\bwebauthn\s+challenge\b", r"\bfido2?\s+challenge\b")),
    ("passkey", "medium", (r"\bpasskeys?\b",)),
    ("relying_party", "high", (r"\brelying\s+party\b", r"\brp\s*id\b", r"\brelying\s+party\s+id\b")),
    ("resident_key", "medium", (r"\bresident\s+keys?\b", r"\bdiscoverable\s+credentials?\b")),
    ("user_verification", "high", (r"\buser\s+verification\b", r"\buv\s+required\b", r"\bverified\s+users?\b")),
)


def detect_query_webauthn_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _WEBAUTHN_CONTEXT_RE.search(text):
        return {"has_webauthn_requirements": False, "requirements": []}
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_webauthn_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
