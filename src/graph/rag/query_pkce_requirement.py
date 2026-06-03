"""Detect PKCE requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PKCE_CONTEXT_RE = re.compile(r"\b(?:pkce|proof\s+key\s+for\s+code\s+exchange|oauth|oidc|authorization\s+code)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("code_challenge", "high", (r"\bcode\s+challenges?\b", r"\bcode_challenge\b")),
    ("code_verifier", "high", (r"\bcode\s+verifiers?\b", r"\bcode_verifier\b")),
    ("native_app_flow", "medium", (r"\bnative\s+app(?:lication)?s?\b", r"\bmobile\s+app(?:lication)?s?\b", r"\bdesktop\s+app(?:lication)?s?\b")),
    ("public_client", "high", (r"\bpublic\s+clients?\b", r"\bclients?\s+without\s+(?:a\s+)?secret\b")),
    ("s256_method", "high", (r"\bs256\b", r"\bsha-?256\b", r"\bplain\s+method\b")),
)


def detect_query_pkce_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _PKCE_CONTEXT_RE.search(text):
        return {"has_pkce_requirements": False, "requirements": []}
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_pkce_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
