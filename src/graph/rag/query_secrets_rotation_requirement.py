"""Detect secret and credential rotation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_ROTATION_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("secret_rotation", re.compile(r"\bsecrets?\s+rotation\b|\brotate\s+(?:the\s+)?secrets?\b", re.I)),
    ("api_key_rotation", re.compile(r"\bapi\s+keys?\s+rotation\b|\brotate\s+(?:the\s+)?api\s+keys?\b", re.I)),
    ("credential_rotation", re.compile(r"\bcredentials?\s+rotation\b|\brotate\s+(?:the\s+)?credentials?\b", re.I)),
    ("password_rotation", re.compile(r"\bpasswords?\s+rotation\b|\brotate\s+(?:the\s+)?passwords?\b", re.I)),
    ("token_rotation", re.compile(r"\btokens?\s+rotation\b|\brotate\s+(?:the\s+)?tokens?\b", re.I)),
    ("expiring_credentials", re.compile(r"\bexpir(?:ing|e[sd]?)\s+(?:credentials?|secrets?|api\s+keys?|passwords?|tokens?)\b|\b(?:credentials?|secrets?|api\s+keys?|passwords?|tokens?)\s+expir(?:e[sd]?|ation|y)\b", re.I)),
    ("automated_rotation", re.compile(r"\b(?:automatic|automated|scheduled)\s+(?:secret|credential|api\s+key|password|token)?\s*rotation\b", re.I)),
)

_CADENCE_RE = re.compile(
    r"\b(?:every|each)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b"
    r"|\b(?:daily|weekly|monthly|quarterly|annually|yearly)\b"
    r"|\b(?:automatic|automated|scheduled)(?:\s+(?:secret|credential|api\s+key|password|token))?\s+rotation\b"
    r"|\bexpir(?:e[sd]?|ing|ation|y)(?:\s+\w+){0,3}\s+(?:after|within|in)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b",
    re.I,
)


def detect_query_secrets_rotation_requirement(query: str) -> list[dict[str, Any]]:
    """Return secret lifecycle rotation requirements mentioned by a query."""

    text = " ".join(str(query or "").split())
    cadence_cues = [
        {"matched_text": match.group(0), "span": [match.start(), match.end()]}
        for match in _CADENCE_RE.finditer(text)
    ]
    rows: list[dict[str, Any]] = []
    for category, pattern in _ROTATION_SPECS:
        for match in pattern.finditer(text):
            rows.append(
                {
                    "category": category,
                    "matched_text": match.group(0),
                    "span": [match.start(), match.end()],
                    "cadence_cues": cadence_cues,
                }
            )
    rows.sort(key=lambda row: (row["span"][0], row["category"]))
    return rows


def detect_query_secrets_rotation_requirements(query: str) -> list[dict[str, Any]]:
    """Alias for callers that use plural requirement naming."""

    return detect_query_secrets_rotation_requirement(query)
