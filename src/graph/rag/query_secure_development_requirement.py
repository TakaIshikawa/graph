"""Detect secure development requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("review", "medium", (r"\bcode\s+reviews?\b", r"\bsecure\s+sdlc\b", r"\bs[-\s]?sdlc\b")),
    ("static_analysis", "high", (r"\bsast\b", r"\bstatic\s+analysis\b")),
    ("dynamic_analysis", "high", (r"\bdast\b", r"\bdynamic\s+analysis\b")),
    ("dependency_scanning", "high", (r"\bdependency\s+scanning\b", r"\bsoftware\s+supply\s+chain\s+scanning\b", r"\bsupply\s+chain\s+scanning\b")),
    ("threat_modeling", "high", (r"\bthreat\s+model(?:ing|ling)?\b",)),
    ("release_gate", "medium", (r"\brelease\s+security\s+gates?\b", r"\bsecurity\s+release\s+gates?\b")),
)


def detect_query_secure_development_requirement(query: str) -> dict[str, Any]:
    """Return secure development requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_secure_development": bool(matches),
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
