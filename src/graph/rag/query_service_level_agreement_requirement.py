"""Detect service-level agreement requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SERVICE_CONTEXT_PATTERNS: tuple[str, ...] = (
    r"\bsaas\b",
    r"\bvendor\b",
    r"\bprovider\b",
    r"\bservice\b",
    r"\bcontract\b",
    r"\bsubscription\b",
    r"\bplatform\b",
    r"\bsla\b",
    r"\bservice[-\s]?level\s+agreement\b",
)
_REQUIREMENT_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("availability_target", "high", (r"\bavailability\s+targets?\b", r"\bavailability\s+commitments?\b", r"\b\d{2,3}(?:\.\d+)?\s*%\s+(?:availability|uptime)\b")),
    ("maintenance_window", "medium", (r"\bmaintenance\s+windows?\b", r"\bscheduled\s+downtime\b", r"\bplanned\s+outages?\b")),
    ("service_credits", "high", (r"\bservice\s+credits?\b", r"\buptime\s+credits?\b", r"\bcredit\s+remed(?:y|ies)\b")),
    ("support_response_time", "high", (r"\bsupport\s+response\s+times?\b", r"\bresponse\s+(?:time|window)\b", r"\btime\s+to\s+respond\b")),
    ("uptime_sla", "high", (r"\buptime\s+sla\b", r"\buptime\s+guarantee\b", r"\buptime\s+commitments?\b", r"\bmonthly\s+uptime\s+percentage\b")),
)


def detect_query_service_level_agreement_requirements(query: str) -> dict[str, Any]:
    """Return service-level agreement requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    requirements = _requirements(text)
    return {
        "has_service_level_agreement_requirements": bool(requirements),
        "requirements": requirements,
    }


def _requirements(text: str) -> list[dict[str, Any]]:
    if not text or not _has_service_context(text):
        return []

    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _REQUIREMENT_SPECS:
        match = _first_match(patterns, text)
        if match:
            rows.append(
                {
                    "category": category,
                    "severity": severity,
                    "matched_text": _matched_text(category, match, text),
                    "span": (match.start(), match.end()),
                }
            )
    return sorted(rows, key=lambda row: row["category"])


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _has_service_context(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in _SERVICE_CONTEXT_PATTERNS)


def _matched_text(category: str, match: re.Match[str], text: str) -> str:
    if category in {"availability_target", "uptime_sla"}:
        target = _availability_target_near(match, text)
        if target:
            return target
    return match.group(0)


def _availability_target_near(match: re.Match[str], text: str) -> str | None:
    window = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)]
    target = re.search(r"\b\d{2,3}(?:\.\d+)?\s*%", window)
    return target.group(0) if target else None
