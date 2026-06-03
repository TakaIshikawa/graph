"""Detect SSO provisioning and deprovisioning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "just_in_time_provisioning",
        "high",
        (
            r"\bjust[-\s]?in[-\s]?time\s+(?:user\s+|account\s+)?provisioning\b",
            r"\bjit\s+(?:user\s+|account\s+)?provisioning\b",
            r"\bprovision\s+(?:users?|accounts?)\s+(?:just[-\s]?in[-\s]?time|on\s+first\s+sso\s+login)\b",
        ),
    ),
    (
        "idp_initiated_user_creation",
        "high",
        (
            r"\bidp[-\s]?initiated\s+(?:user\s+|account\s+)?creation\b",
            r"\bidentity\s+provider[-\s]?initiated\s+(?:user\s+|account\s+)?creation\b",
            r"\bcreate\s+(?:users?|accounts?)\s+from\s+(?:the\s+)?idp\b",
            r"\bcreate\s+(?:users?|accounts?)\s+after\s+(?:sso|single\s+sign[-\s]?on)\s+login\b",
        ),
    ),
    (
        "automatic_account_provisioning",
        "high",
        (
            r"\bautomatic\s+(?:user\s+|account\s+)?provisioning\b",
            r"\bautomated?\s+(?:user\s+|account\s+)?provisioning\b",
            r"\bauto[-\s]?provision(?:ing)?\s+(?:users?|accounts?)\b",
            r"\bprovision\s+(?:users?|accounts?)\s+automatically\b",
        ),
    ),
    (
        "deprovisioning",
        "high",
        (
            r"\bde[-\s]?provision(?:ing)?\b",
            r"\bde[-\s]?provision\s+(?:users?|accounts?)\b",
            r"\bremove\s+(?:users?|accounts?)\s+(?:when|after)\s+(?:they\s+)?leave\b",
            r"\bdisable\s+(?:users?|accounts?)\s+(?:automatically|on\s+termination)\b",
            r"\b(?:user|account)\s+offboarding\b",
        ),
    ),
    (
        "account_lifecycle",
        "medium",
        (
            r"\baccount\s+lifecycle\b",
            r"\buser\s+lifecycle\b",
            r"\bidentity\s+lifecycle\b",
            r"\bjoiner[-\s]?mover[-\s]?leaver\b",
            r"\bjml\s+(?:process|workflow|lifecycle)\b",
        ),
    ),
)


def detect_query_sso_provisioning_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_sso_provisioning_requirement": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
