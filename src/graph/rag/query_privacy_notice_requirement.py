"""Detect privacy notice and user-facing disclosure requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("privacy_notice", (r"\bprivacy\s+notice\b",)),
    ("privacy_policy_disclosure", (r"\bprivacy\s+policy\s+disclosure\b", r"\bprivacy\s+policy\b")),
    ("notice_at_collection", (r"\bnotice\s+at\s+collection\b", r"\bcollection\s+notice\b")),
    ("purpose_disclosure", (r"\bpurpose\s+disclosure\b", r"\bdisclose\s+purposes?\b")),
    ("user_facing_notice", (r"\buser-facing\s+(?:data-use\s+)?notice\b", r"\bdata-use\s+notice\b")),
)


def detect_query_privacy_notice_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {"requires_privacy_notice": bool(categories), "cue_categories": categories}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
