"""Detect SIEM integration requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("siem_export", "high", re.compile(r"\b(?:siem\s+(?:export|integration|connector)|export\s+(?:to|for)\s+siem)\b", re.I)),
    ("splunk", "high", re.compile(r"\bsplunk\b", re.I)),
    ("sentinel", "high", re.compile(r"\b(?:microsoft\s+sentinel|azure\s+sentinel)\b", re.I)),
    ("qradar", "high", re.compile(r"\b(?:ibm\s+)?qradar\b", re.I)),
    ("syslog_forwarding", "high", re.compile(r"\b(?:syslog|syslog\s+forwarding|forward\s+(?:events?|logs?)\s+via\s+syslog)\b", re.I)),
    ("security_event_streaming", "high", re.compile(r"\b(?:security\s+event\s+(?:streaming|stream|feed)|stream\s+security\s+events?)\b", re.I)),
    ("alert_enrichment", "medium", re.compile(r"\b(?:alert\s+enrichment|enrich(?:ed)?\s+alerts?|alert\s+context)\b", re.I)),
)


def detect_query_siem_integration_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for category, severity, pattern in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "severity": severity, "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
