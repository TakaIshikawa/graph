"""Detect business continuity requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("continuity_plan", "high", (r"\bbusiness\s+continuity\b", r"\bbcp\b", r"\bcontinuity\s+plans?\b")),
    ("testing", "medium", (r"\bcontinuity\s+testing\b", r"\bbcp\s+testing\b")),
    ("crisis_management", "high", (r"\bcrisis\s+management\b",)),
    ("alternate_worksite", "medium", (r"\balternate\s+worksites?\b", r"\balternative\s+worksites?\b")),
    ("recovery_objective", "high", (r"\brto\b", r"\brecovery\s+time\s+objectives?\b")),
    ("ownership", "medium", (r"\bcontinuity\s+plan\s+owners?\b", r"\bbcp\s+owners?\b", r"\bplan\s+ownership\b")),
)


def detect_query_business_continuity_requirement(query: str) -> dict[str, Any]:
    """Return business continuity requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_business_continuity": bool(matches),
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
