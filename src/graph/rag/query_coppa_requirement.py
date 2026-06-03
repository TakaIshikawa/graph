"""Detect COPPA requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("coppa", "high", (r"\bcoppa\b", r"\bchildren'?s\s+online\s+privacy\s+protection\s+act\b")),
    ("under_13", "high", (r"\bunder\s+13\b", r"\bchildren\s+under\s+thirteen\b", r"\busers?\s+under\s+13\b")),
    ("verifiable_parental_consent", "high", (r"\bverifiable\s+parental\s+consent\b",)),
    ("parental_notice", "medium", (r"\bparental\s+notice\b", r"\bnotice\s+to\s+parents\b")),
    ("child_directed_service", "medium", (r"\bchild-directed\s+(?:service|website|app)\b", r"\bdirected\s+to\s+children\b")),
    ("deletion_rights", "medium", (r"\bchild(?:ren)?'?s\s+data\s+deletion\b", r"\bdelete\s+child(?:ren)?'?s\s+data\b")),
)


def detect_query_coppa_requirement(query: str) -> dict[str, Any]:
    matches = _matches(query)
    categories = sorted(dict.fromkeys(match["category"] for match in matches))
    return {"requires_coppa": bool(matches), "categories": categories, "matches": matches}


def _matches(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": match.span()})
    return sorted(rows, key=lambda row: (row["category"], row["span"][0], row["matched_text"].casefold()))
