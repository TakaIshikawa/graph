"""Detect SCIM lifecycle and provisioning requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "scim",
        "high",
        (
            r"\bscim\b",
            r"\bsystem\s+for\s+cross[-\s]?domain\s+identity\s+management\b",
        ),
    ),
    (
        "user_provisioning",
        "high",
        (
            r"\buser\s+provisioning\b",
            r"\baccount\s+provisioning\b",
            r"\bprovision\s+(?:users?|accounts?)\b",
            r"\bautomated?\s+(?:user\s+|account\s+)?provisioning\b",
            r"\bautomatic\s+(?:user\s+|account\s+)?provisioning\b",
        ),
    ),
    (
        "group_provisioning",
        "medium",
        (
            r"\bgroup\s+provisioning\b",
            r"\bprovision\s+groups?\b",
            r"\bgroup\s+membership\s+provisioning\b",
            r"\bprovision\s+group\s+memberships?\b",
        ),
    ),
    (
        "deprovisioning",
        "high",
        (
            r"\bde[-\s]?provision(?:ing)?\b",
            r"\bdeprovision\s+(?:users?|accounts?)\b",
            r"\bremove\s+(?:users?|accounts?)\s+automatically\b",
            r"\bdisable\s+(?:users?|accounts?)\s+automatically\b",
            r"\bautomated?\s+(?:user\s+|account\s+)?offboarding\b",
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
)


def detect_scim_requirement(query: str) -> dict[str, Any]:
    """Return SCIM provisioning requirements mentioned by a query."""
    text = _normalize_query(query)
    matches = _detect_matches(text)
    return {
        "requires_scim": bool(matches),
        "categories": [match["category"] for match in matches],
        "matches": matches,
    }


def _detect_matches(text: str) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, (category, severity, patterns) in enumerate(_REQUIREMENTS):
        found = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if found:
            match = min(found, key=lambda item: item.start())
            rows.append(
                (
                    match.start(),
                    index,
                    {
                        "category": category,
                        "severity": severity,
                        "matched_text": match.group(0),
                        "span": (match.start(), match.end()),
                        "snippet": _snippet(text, match.start(), match.end()),
                    },
                )
            )
    return [row for _start, _index, row in sorted(rows, key=lambda item: (item[0], item[1]))]


def _snippet(text: str, start: int, end: int, radius: int = 40) -> str:
    snippet_start = max(0, start - radius)
    snippet_end = min(len(text), end + radius)
    return text[snippet_start:snippet_end].strip()


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
