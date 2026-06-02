"""Detect access review requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cadence", "medium", (r"\b(?:quarterly|annual|annually|monthly|semiannual|semi-annual)\s+access\s+reviews?\b",)),
    ("access_review", "medium", (r"\baccess\s+reviews?\b",)),
    ("access_recertification", "high", (r"\baccess\s+recertification\b", r"\buser\s+access\s+recertification\b")),
    ("entitlement_review", "medium", (r"\bentitlement\s+reviews?\b",)),
    ("role_review", "medium", (r"\brole\s+reviews?\b",)),
    ("least_privilege_review", "high", (r"\bleast\s+privilege\s+reviews?\b",)),
    ("reviewer_signoff", "medium", (r"\breviewer\s+sign[-\s]?off\b", r"\breviewer\s+approval\b")),
)


def detect_query_access_review_requirement(query: str) -> dict[str, Any]:
    """Return access review requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_access_review": bool(matches),
        "categories": [match["category"] for match in matches],
        "matches": matches,
    }


def _detect_matches(text: str) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, (category, severity, patterns) in enumerate(_CATEGORY_SPECS):
        found = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if found:
            match = min(found, key=lambda item: item.start())
            rows.append((match.start(), index, {"category": category, "severity": severity, "matched_text": match.group(0), "span": (match.start(), match.end())}))
    return [row for _start, _index, row in sorted(rows, key=lambda item: (item[0], item[1]))]
