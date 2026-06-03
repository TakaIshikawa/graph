"""Detect OpenID Connect requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_OIDC_CONTEXT_RE = re.compile(r"\b(?:oidc|open\s*id\s+connect|openid\s+connect)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("claims_mapping", "medium", (r"\bclaims?\s+mapping\b", r"\bmap\s+(?:oidc\s+)?claims?\b", r"\bclaim\s+mappings?\b")),
    ("id_token", "high", (r"\bid\s+tokens?\b", r"\bid_tokens?\b")),
    ("issuer_discovery", "high", (r"\bissuer\b", r"\bdiscovery\s+documents?\b", r"\bopenid[-\s]?configuration\b", r"\bwell-known\s+configuration\b")),
    ("jwks", "high", (r"\bjwks\b", r"\bjson\s+web\s+key\s+sets?\b", r"\bjwks_uri\b", r"\bsigning\s+keys?\b")),
    ("nonce_state", "high", (r"\bnonce\b", r"\bstate\s+validation\b", r"\bvalidate\s+(?:nonce|state)\b", r"\bnonce\s+and\s+state\b")),
)


def detect_query_oidc_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    if not _OIDC_CONTEXT_RE.search(text):
        return {"has_oidc_requirements": False, "requirements": []}

    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_oidc_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
