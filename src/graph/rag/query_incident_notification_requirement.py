"""Detect security incident notification requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_INCIDENT_CONTEXT_RE = re.compile(r"\b(?:security\s+incident|data\s+breach|breach|incident\s+notification|regulatory\s+notice)\b", re.I)
_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("notification_timeline", "high", (r"\b(?:notify|notification|notice)(?:\s+\w+){0,3}\s+(?:within|in|by)\s+\d+\s+(?:hours?|days?)\b", r"\bnotification\s+timelines?\b")),
    ("notification_channel", "medium", (r"\bnotification\s+channels?\b", r"\b(?:email|portal|webhook)\s+(?:notification|notice)s?\b")),
    ("severity_threshold", "high", (r"\bseverity\s+thresholds?\b", r"\b(?:critical|high)\s+severity\s+incidents?\b")),
    ("breach_regulatory_notice", "high", (r"\bregulatory\s+notice\b", r"\bbreach\s+notification\b", r"\bdata\s+breach\b")),
    ("status_updates", "medium", (r"\bstatus\s+updates?\b", r"\bongoing\s+updates?\b")),
    ("root_cause_report", "medium", (r"\broot[-\s]cause\s+(?:analysis|report)\b", r"\brca\s+report\b", r"\bpost[-\s]incident\s+report\b")),
    ("customer_contact", "medium", (r"\bcustomer\s+contacts?\b", r"\bwho\s+gets\s+notified\b", r"\bnotify\s+customers?\b")),
)


def detect_query_incident_notification_requirement(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not _INCIDENT_CONTEXT_RE.search(text):
        return []
    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"category": category, "severity": severity, "matched_text": match.group(0), "evidence": match.group(0)})
    return sorted(rows, key=lambda row: row["category"])
