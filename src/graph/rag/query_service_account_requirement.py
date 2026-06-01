"""Detect service-account requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_IDENTITY_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bot_account", (r"\bbot\s+accounts?\b", r"\bautomation\s+accounts?\b")),
    ("credential_owner", (r"\bcredential\s+owners?\b", r"\bowner\s+for\s+(?:api\s+)?credentials?\b")),
    ("machine_user", (r"\bmachine\s+users?\b",)),
    ("non_human_identity", (r"\bnon[-\s]?human\s+identit(?:y|ies)\b", r"\bnhi\b")),
    ("service_account", (r"\bservice\s+accounts?\b",)),
    ("workload_identity", (r"\bworkload\s+identit(?:y|ies)\b",)),
)
_ROTATION = (r"\bcredential\s+rotation\b", r"\brotate\s+(?:the\s+)?credentials?\b", r"\brotation\s+schedule\b")


def detect_query_service_account_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    identity_types = []
    matched_cues = []
    for category, patterns in _IDENTITY_TYPES:
        match = _first_match(patterns, text)
        if match:
            identity_types.append(category)
            matched_cues.append({"category": category, "matched_text": match.group(0)})
    rotation_mentioned = _matches_any(_ROTATION, text)

    requires_service_account = bool(identity_types)
    return {
        "requires_service_account": requires_service_account,
        "identity_types": identity_types,
        "rotation_mentioned": rotation_mentioned,
        "matched_cues": matched_cues,
        "severity": "high" if rotation_mentioned and requires_service_account else ("medium" if requires_service_account else "none"),
    }


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
