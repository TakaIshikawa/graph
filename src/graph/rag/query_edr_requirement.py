"""Detect endpoint detection and response requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\bedr\b"
    r"|\bendpoint\s+detection\s+and\s+response\b"
    r"|\bendpoint\s+telemetry\b"
    r"|\bbehavioral\s+detection\b"
    r"|\bisolat(?:e|ion)\b"
    r"|\bquarantine\b"
    r"|\bagent\s+coverage\b"
    r"|\bthreat\s+hunting\b",
    re.I,
)

_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("endpoint_coverage", re.compile(r"\bagent\s+coverage\b|\bendpoint\s+coverage\b|\bcovered\s+endpoints?\b|\bworkstation(?:s)?\b|\bservers?\b", re.I)),
    ("telemetry", re.compile(r"\bendpoint\s+telemetry\b|\bprocess\s+telemetry\b|\bfile\s+events?\b|\bnetwork\s+events?\b|\bcommand[-\s]?line\b", re.I)),
    ("detection", re.compile(r"\bedr\b|\bendpoint\s+detection\s+and\s+response\b|\bbehavioral\s+detection\b|\bmalware\s+detection\b|\bsuspicious\s+behavior\b", re.I)),
    ("response_actions", re.compile(r"\bisolat(?:e|ion)\b|\bquarantine\b|\bkill\s+process(?:es)?\b|\bremediat(?:e|ion)\b|\bcontainment\b", re.I)),
    ("threat_hunting", re.compile(r"\bthreat\s+hunting\b|\bhunt\s+queries\b|\binvestigation\s+queries\b", re.I)),
    ("alert_integration", re.compile(r"\balert\s+integration\b|\balerts?\b|\bsiem\b|\bsoc\b|\bwebhooks?\b|\bnotifications?\b", re.I)),
)


def detect_query_edr_requirements(query: str) -> dict[str, Any]:
    """Return EDR requirement categories mentioned by a query."""

    text = _normalize_query(query)
    if not _CONTEXT_RE.search(text):
        return {"has_edr_requirements": False, "requirements": []}

    rows: list[dict[str, Any]] = []
    for category, pattern in _SPECS:
        match = pattern.search(text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: row["category"])
    return {"has_edr_requirements": bool(rows), "requirements": rows}


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
