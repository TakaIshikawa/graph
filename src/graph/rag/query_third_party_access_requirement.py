"""Detect third-party access requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("vendor_access", "high", (r"\bvendor\s+access\b", r"\bthird[-\s]party\s+access\b")),
    ("contractor_access", "high", (r"\bcontractor\s+access\b",)),
    ("support_access", "medium", (r"\bsupport\s+engineer\s+access\b", r"\bsupport\s+access\b")),
    ("external_admin", "high", (r"\bexternal\s+admin\s+access\b", r"\bexternal\s+administrator\s+access\b")),
    ("just_in_time_access", "high", (r"\bjust[-\s]in[-\s]time\s+third[-\s]party\s+access\b", r"\bjit\s+third[-\s]party\s+access\b", r"\bjit\s+vendor\s+access\b")),
)


def detect_query_third_party_access_requirement(query: str) -> dict[str, Any]:
    """Return third-party access requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_third_party_access": bool(matches),
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
