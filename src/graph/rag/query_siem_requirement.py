"""Detect SIEM and security-event forwarding requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("siem_integration", "high", (r"\bsiem\b", r"\bsecurity\s+information\s+and\s+event\s+management\b")),
    ("log_forwarding", "high", (r"\blog\s+forwarding\b", r"\bforward\s+(?:security\s+)?logs?\b")),
    (
        "event_export",
        "high",
        (
            r"\bsecurity\s+events?\s+export\b",
            r"\bexport\s+security\s+events?\b",
            r"\bevent\s+export\s+(?:to|for)\s+(?:a\s+)?siem\b",
        ),
    ),
    ("syslog", "high", (r"\bsyslog\b",)),
    ("splunk", "high", (r"\bsplunk\b",)),
    ("sentinel", "high", (r"\b(?:microsoft\s+sentinel|azure\s+sentinel)\b",)),
    ("qradar", "high", (r"\b(?:ibm\s+)?qradar\b",)),
    (
        "security_event_streaming",
        "high",
        (
            r"\bsecurity\s+events?\s+(?:streaming|stream|feed)\b",
            r"\bstream\s+security\s+events?\b",
        ),
    ),
)


def detect_query_siem_requirement(query: str) -> dict[str, Any]:
    """Return SIEM integration and security-event forwarding needs in a query."""
    text = _normalize_query(query)
    matches = _detect_matches(text)
    return {
        "requires_siem": bool(matches),
        "categories": [match["category"] for match in matches],
        "matches": matches,
        "recommendations": ["retrieve SIEM integration and security event export documentation"] if matches else [],
        "confidence": "high" if matches else "none",
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
                        "span": [match.start(), match.end()],
                    },
                )
            )
    return [row for _start, _index, row in sorted(rows, key=lambda item: (item[0], item[1]))]


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
