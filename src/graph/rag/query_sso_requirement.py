"""Detect single sign-on requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PROVIDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("azure_ad", (r"\bazure\s+ad\b", r"\bentra\s+id\b", r"\bmicrosoft\s+entra\b")),
    ("google_workspace", (r"\bgoogle\s+workspace\b", r"\bgoogle\s+(?:login|sso|identity)\b")),
    ("okta", (r"\bokta\b",)),
)
_PROTOCOLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("oidc", (r"\boidc\b", r"\bopenid\s+connect\b")),
    ("saml", (r"\bsaml\b", r"\bsaml\s*2(?:\.0)?\b")),
)
_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sso", (r"\bsingle\s+sign[-\s]?on\b", r"\bsso\b")),
    ("identity_provider", (r"\bidentity[-\s]?provider\s+login\b", r"\bidp\s+login\b", r"\blogin\s+through\s+(?:an\s+)?identity\s+provider\b")),
)


def detect_query_sso_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    providers = [name for name, patterns in _PROVIDERS if _matches_any(patterns, text)]
    protocols = [name for name, patterns in _PROTOCOLS if _matches_any(patterns, text)]
    matched_cues = []
    for category, patterns in _CUES:
        match = _first_match(patterns, text)
        if match:
            matched_cues.append({"category": category, "matched_text": match.group(0)})

    requires_sso = bool(providers or protocols or matched_cues)
    return {
        "requires_sso": requires_sso,
        "providers": providers,
        "protocols": protocols,
        "matched_cues": matched_cues,
        "confidence": "high" if protocols or matched_cues else ("medium" if providers else "none"),
    }


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
