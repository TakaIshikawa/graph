"""Detect deduplication requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "duplicate_suppression",
        (
            r"\bduplicate\s+suppression\b",
            r"\bsuppress(?:ing|ion)?\s+duplicates?\b",
            r"\bprevent\s+duplicates?\b",
            r"\bavoid\s+duplicates?\b",
            r"\bdiscard\s+duplicates?\b",
            r"\bdrop\s+duplicates?\b",
            r"\bde[-\s]?dupe\b",
            r"\bdeduplicat(?:e|es|ed|ing|ion)\b",
        ),
    ),
    (
        "dedupe_key",
        (
            r"\bde[-\s]?dupe\s+keys?\b",
            r"\bdeduplication\s+keys?\b",
            r"\bdeduplicate\s+by\s+(?:key|id|field|fields|identifier)\b",
            r"\bunique\s+(?:dedupe\s+)?keys?\b",
            r"\bduplicate\s+(?:detection\s+)?keys?\b",
        ),
    ),
    (
        "content_hash",
        (
            r"\bcontent\s+hash(?:es|ing)?\b",
            r"\bhash(?:ing)?\s+(?:the\s+)?(?:content|payload|body|record)\b",
            r"\bpayload\s+hash(?:es|ing)?\b",
            r"\bbody\s+hash(?:es|ing)?\b",
            r"\brecord\s+hash(?:es|ing)?\b",
        ),
    ),
    (
        "idempotency_token",
        (
            r"\bidempotency\s+(?:tokens?|keys?)\b",
            r"\bidempotent\s+(?:tokens?|keys?)\b",
            r"\brequest\s+idempotency\b",
        ),
    ),
    (
        "duplicate_merge",
        (
            r"\bmerge\s+duplicate\s+records?\b",
            r"\bmerge\s+duplicates?\b",
            r"\bduplicate\s+(?:record\s+)?merg(?:e|es|ing)\b",
            r"\bcoalesc(?:e|ing)\s+duplicates?\b",
            r"\bconsolidat(?:e|ing)\s+duplicate\s+records?\b",
        ),
    ),
)

_DEDUPE_CONTEXT = (
    r"\bde[-\s]?dupe\b",
    r"\bdeduplicat(?:e|es|ed|ing|ion)\b",
    r"\bduplicates?\b",
)


def detect_query_deduplication_requirement(query: str) -> dict[str, Any]:
    """Return duplicate handling requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    rows = _requirement_rows(text)
    categories = [row["category"] for row in rows]
    has_deduplication_requirement = bool(rows)

    return {
        "has_deduplication_requirement": has_deduplication_requirement,
        "requirements": rows,
        "categories": categories,
        "matched_phrases": [row["matched_text"] for row in rows],
        "confidence": _confidence(categories, text),
    }


def _requirement_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, patterns in _CATEGORY_SPECS:
        match = _first_match(patterns, text)
        if match and _is_contextual_match(category, text):
            rows.append({"category": category, "matched_text": match.group(0), "span": (match.start(), match.end())})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"]))


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _is_contextual_match(category: str, text: str) -> bool:
    if category != "idempotency_token":
        return True
    return any(re.search(pattern, text, re.I) for pattern in _DEDUPE_CONTEXT)


def _confidence(categories: list[str], text: str) -> str:
    if not categories:
        return "none"
    if len(categories) >= 2:
        return "high"
    if "idempotency_token" in categories and re.search(r"\bduplicates?\b|\bdedupe\b|\bdeduplication\b", text, re.I):
        return "medium"
    return "medium"
