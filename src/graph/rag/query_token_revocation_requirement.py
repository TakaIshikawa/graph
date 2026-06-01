"""Detect token revocation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_INTENT_RE = re.compile(r"\b(?:revok(?:e|ed|ing|ation)|invalidat(?:e|ed|ion)|denylist|blocklist|logout|introspection|revocation\s+endpoint)\b", re.I)
_TOKEN_RE = re.compile(r"\btokens?\b|\boauth\b|\boidc\b|\bjwt\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("refresh_token", "high", (r"\brefresh\s+tokens?\b",)),
    ("access_token", "high", (r"\baccess\s+tokens?\b",)),
    ("logout_invalidation", "medium", (r"\blogout\s+invalidat(?:e|ion)\b", r"\binvalidat(?:e|ion)\s+on\s+logout\b", r"\brevoke\s+on\s+logout\b")),
    ("denylist_blocklist", "medium", (r"\bdenylist\b", r"\bblocklist\b", r"\brevocation\s+list\b")),
    ("introspection", "medium", (r"\bintrospection\b", r"\btoken\s+introspection\b")),
    ("revocation_endpoint", "high", (r"\brevocation\s+endpoint\b", r"\boauth\s+revocation\b", r"\b/token/revoke\b")),
)


def detect_query_token_revocation_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = []
    if _INTENT_RE.search(text) and _TOKEN_RE.search(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_token_revocation_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
