"""Detect key management requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("lifecycle", "high", (r"\bkey\s+management\b", r"\bkms\b", r"\bkey\s+rotation\b", r"\bkey\s+destruction\b")),
    ("custody", "high", (r"\bkey\s+custody\b", r"\bcustody\s+of\s+(?:the\s+)?keys?\b")),
    ("hardware_security_module", "high", (r"\bhsm\b", r"\bhardware\s+security\s+modules?\b")),
    ("envelope_encryption", "medium", (r"\benvelope\s+encryption\b",)),
)


def detect_query_key_management_requirement(query: str) -> dict[str, Any]:
    """Return key management requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_key_management": bool(matches),
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
