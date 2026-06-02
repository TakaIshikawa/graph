"""Detect change management requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("change_management", "medium", (r"\bchange\s+management\b", r"\bchange\s+control\s+process\b")),
    ("cab_approval", "high", (r"\bcab\s+approval\b", r"\bchange\s+advisory\s+board\s+approval\b")),
    ("change_ticket", "medium", (r"\bchange\s+tickets?\b", r"\bchange\s+requests?\b")),
    ("change_freeze", "high", (r"\bchange\s+freezes?\b", r"\bfreeze\s+window\b")),
    ("emergency_change", "high", (r"\bemergency\s+changes?\b", r"\bexpedited\s+changes?\b")),
    ("production_change_control", "high", (r"\bproduction\s+change\s+controls?\b", r"\bprod(?:uction)?\s+change\s+approval\b")),
)


def detect_query_change_management_requirement(query: str) -> dict[str, Any]:
    """Return change management requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_change_management": bool(matches),
        "categories": [match["category"] for match in matches],
        "matches": matches,
    }


def _detect_matches(text: str) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, (category, severity, patterns) in enumerate(_CATEGORY_SPECS):
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
                    },
                )
            )
    return [row for _start, _index, row in sorted(rows, key=lambda item: (item[0], item[1]))]
