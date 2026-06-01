"""Detect SCIM provisioning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "automated_provisioning",
        "high",
        (
            r"\bautomated?\s+(?:user\s+|account\s+)?provisioning\b",
            r"\bautomatic\s+(?:user\s+|account\s+)?provisioning\b",
            r"\b(?:user|account)\s+provisioning\b",
            r"\bprovision\s+(?:users?|accounts?)\b",
            r"\bjust[-\s]?in[-\s]?time\s+provisioning\b",
            r"\bjit\s+provisioning\b",
        ),
    ),
    (
        "deprovisioning",
        "high",
        (
            r"\bde[-\s]?provision(?:ing)?\b",
            r"\bdeprovision\s+(?:users?|accounts?)\b",
            r"\bremove\s+(?:users?|accounts?)\s+automatically\b",
            r"\bautomated?\s+(?:user\s+|account\s+)?offboarding\b",
            r"\bdisable\s+(?:users?|accounts?)\s+automatically\b",
        ),
    ),
    (
        "directory_sync",
        "medium",
        (
            r"\bdirectory\s+sync(?:hronization)?\b",
            r"\bsync\s+(?:users?|accounts?)\s+from\s+(?:the\s+)?directory\b",
            r"\bactive\s+directory\s+sync\b",
            r"\bldap\s+sync\b",
            r"\b(?:entra\s+id|azure\s+ad|google\s+workspace|okta)\s+directory\s+sync\b",
        ),
    ),
    (
        "group_sync",
        "medium",
        (
            r"\bgroup\s+sync(?:hronization)?\b",
            r"\bsync\s+groups?\b",
            r"\bgroup\s+membership\s+sync\b",
            r"\bsync\s+group\s+memberships?\b",
            r"\bmap\s+(?:idp\s+)?groups?\b",
        ),
    ),
    (
        "identity_lifecycle",
        "medium",
        (
            r"\bidentity\s+lifecycle\b",
            r"\buser\s+lifecycle\s+management\b",
            r"\bjoiner[-\s]?mover[-\s]?leaver\b",
            r"\bjml\s+(?:process|workflow|lifecycle)\b",
        ),
    ),
    (
        "scim",
        "high",
        (
            r"\bscim\b",
            r"\bsystem\s+for\s+cross[-\s]?domain\s+identity\s+management\b",
        ),
    ),
)


def detect_query_scim_provisioning_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    requirements = []
    for category, severity, patterns in _REQUIREMENTS:
        match = _first_match(patterns, text)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_scim_provisioning_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
