"""Detect uptime and availability SLA requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_COMMITMENT = (
    r"\buptime\s+sla\b",
    r"\bavailability\s+targets?\b",
    r"\bservice\s+availability\b",
    r"\bmonthly\s+uptime\s+percentage\b",
    r"\bdowntime\s+allowance\b",
    r"\bmaintenance\s+exclusions?\b",
)
_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("availability_target", (r"\bavailability\s+targets?\b", r"\bservice\s+availability\b")),
    ("monthly_uptime_percentage", (r"\bmonthly\s+uptime\s+percentage\b",)),
    ("downtime_allowance", (r"\bdowntime\s+allowance\b",)),
    ("maintenance_exclusion", (r"\bmaintenance\s+exclusions?\b",)),
)


def detect_query_uptime_sla_requirement(query: str) -> dict[str, Any]:
    text = " ".join(str(query or "").split())
    percentages = re.findall(r"\b\d{2,3}(?:\.\d+)?\s*%", text)
    matched = _matches(text, _COMMITMENT)
    if percentages and re.search(r"\b(?:uptime|availability|sla)\b", text, re.I):
        matched.extend(percentages)
    cues = [name for name, patterns in _CUES if _matches(text, patterns)]
    return {
        "requires_uptime_sla": bool(matched),
        "matched_phrases": sorted(dict.fromkeys(matched), key=str.casefold),
        "percentage_targets": percentages,
        "cue_categories": cues,
        "confidence": "high" if matched else "none",
    }


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [match.group(0) for pattern in patterns for match in re.finditer(pattern, text, re.I)]
