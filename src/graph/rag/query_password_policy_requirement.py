"""Detect password policy requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("breach_screening", re.compile(r"\b(?:breached\s+password|password\s+(?:breach|screening)|compromised\s+password|have\s+i\s+been\s+pwned)\b", re.I), "high"),
    ("complexity", re.compile(r"\b(?:password\s+complexity|complex\s+password|uppercase|lowercase|special\s+characters?|symbols?\s+in\s+password)\b", re.I), "medium"),
    ("length", re.compile(r"\b(?:password\s+(?:minimum\s+)?length|min(?:imum)?[-\s]?\d+\s+characters?|at\s+least\s+\d+\s+characters?)\b", re.I), "medium"),
    ("lockout", re.compile(r"\b(?:account\s+lockout|password\s+lockout|failed\s+login\s+attempts?|lock\s+after\s+\d+)\b", re.I), "high"),
    ("reuse_history", re.compile(r"\b(?:password\s+(?:reuse|history)|previous\s+passwords?|remember\s+\d+\s+passwords?)\b", re.I), "medium"),
    ("rotation", re.compile(r"\b(?:password\s+(?:rotation|expiry|expiration)|rotation|rotate\s+passwords?|change\s+passwords?\s+every)\b", re.I), "medium"),
)


def detect_query_password_policy_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    if "password" not in normalized.casefold() and not re.search(r"\b(?:uppercase|lowercase|lock\s+after|failed\s+login)\b", normalized, re.I):
        return []
    rows = []
    for category, pattern, severity in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "severity": severity})
    rows.sort(key=lambda row: row["category"])
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
