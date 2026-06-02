"""Detect data classification requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("classification_scheme", "medium", (r"\bdata\s+classification\b", r"\bclassification\s+scheme\b")),
    ("sensitive_data", "high", (r"\bconfidential\s+data\b", r"\brestricted\s+data\b", r"\bpii\s+sensitivity\b", r"\bsensitive\s+data\b")),
    ("regulated_data", "high", (r"\bregulated\s+data\b", r"\bregulated\s+data\s+handling\b")),
    ("handling_label", "medium", (r"\bclassification\s+labels?\b", r"\bhandling\s+labels?\b", r"\bdata\s+handling\s+labels?\b")),
)


def detect_query_data_classification_requirement(query: str) -> dict[str, Any]:
    """Return data classification requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_data_classification": bool(matches),
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
