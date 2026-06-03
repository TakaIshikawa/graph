"""Detect FERPA requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ferpa", "high", (r"\bferpa\b", r"\bfamily\s+educational\s+rights\s+and\s+privacy\s+act\b")),
    ("education_records", "high", (r"\beducation\s+records?\b", r"\bstudent\s+records?\b")),
    ("student_pii", "high", (r"\bstudent\s+pii\b", r"\bpersonally\s+identifiable\s+information\s+from\s+students?\b")),
    ("directory_information", "medium", (r"\bdirectory\s+information\b",)),
    ("consent", "medium", (r"\bparent(?:al)?\s+consent\b", r"\bstudent\s+consent\b", r"\bschool\s+official\s+exceptions?\b")),
    ("access_amendment", "medium", (r"\beligible\s+student\s+access\b", r"\baccess\s+and\s+amend(?:ment)?\b", r"\bamend\s+education\s+records?\b")),
)


def detect_query_ferpa_requirement(query: str) -> dict[str, Any]:
    matches = _matches(query)
    categories = sorted(dict.fromkeys(match["category"] for match in matches))
    return {"requires_ferpa": bool(matches), "categories": categories, "matches": matches}


def _matches(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": match.span()})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"], row["matched_text"].casefold()))
