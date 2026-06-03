"""Detect OAuth DPoP requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "dpop",
        "dpop_specific",
        "high",
        (r"\bdpop\b", r"\bdemonstrating\s+proof[-\s]?of[-\s]?possession\b"),
    ),
    (
        "dpop_proof_jwt",
        "dpop_specific",
        "high",
        (r"\bdpop\s+proof\s+jwts?\b", r"\bdpop\s+jwts?\b", r"\bproof\s+jwts?\b"),
    ),
    (
        "jkt_thumbprint",
        "dpop_specific",
        "high",
        (r"\bjkt\s+thumbprints?\b", r"\bjwk\s+thumbprints?\b", r"\bcnf\.jkt\b", r"\bjkt\b"),
    ),
    (
        "proof_of_possession_token",
        "proof_of_possession",
        "high",
        (r"\bproof[-\s]?of[-\s]?possession\s+tokens?\b", r"\bpop\s+tokens?\b"),
    ),
    (
        "sender_constrained_token",
        "proof_of_possession",
        "high",
        (r"\bsender[-\s]?constrained\s+tokens?\b", r"\bsender[-\s]?constraint\b"),
    ),
    (
        "token_binding",
        "proof_of_possession",
        "medium",
        (r"\btoken\s+binding\b", r"\bbound\s+tokens?\b", r"\bbind\s+(?:access\s+)?tokens?\b"),
    ),
)


def detect_dpop_requirement(query: str) -> dict[str, Any]:
    """Return OAuth DPoP and proof-of-possession requirements mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    for category, trigger_type, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append(
                {
                    "category": category,
                    "matched_text": match.group(0),
                    "trigger_type": trigger_type,
                    "severity": severity,
                }
            )
    requirements.sort(key=lambda row: row["category"])
    return {"has_dpop_requirement": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
