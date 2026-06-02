"""Detect consent-management evidence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("consent_collection", (r"\bcollect(?:ion)?\s+consent\b", r"\bconsent\s+collection\b")),
    ("opt_in", (r"\bopt-?in\b", r"\baffirmative\s+consent\b")),
    ("opt_out", (r"\bopt-?out\b", r"\bdo\s+not\s+sell\b")),
    ("consent_withdrawal", (r"\bwithdraw(?:al)?\s+consent\b", r"\bconsent\s+withdrawal\b", r"\brevoke\s+consent\b")),
    ("cookie_consent", (r"\bcookie\s+consent\b", r"\bcookie\s+banner\b")),
    ("marketing_consent", (r"\bmarketing\s+consent\b", r"\bemail\s+consent\b")),
    ("purpose_consent", (r"\bpurpose\s+consent\b", r"\bconsent\s+by\s+purpose\b")),
    ("consent_records", (r"\bconsent\s+records?\b", r"\bconsent\s+log\b")),
)
_FRAMEWORKS: tuple[tuple[str, str], ...] = (
    ("GDPR", r"\bgdpr\b"),
    ("CCPA", r"\bccpa\b"),
    ("CMP", r"\bcmp\b|\bconsent\s+management\s+platform\b"),
    ("IAB TCF", r"\biab\s+tcf\b|\btcf\b"),
    ("GPC", r"\bgpc\b|\bglobal\s+privacy\s+control\b"),
)


def detect_query_consent_management_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {
        "requires_consent_management": bool(categories),
        "cue_categories": categories,
        "frameworks": _framework_mentions(text) if categories else [],
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _framework_mentions(text: str) -> list[str]:
    return [name for name, pattern in _FRAMEWORKS if re.search(pattern, text, re.I)]


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
