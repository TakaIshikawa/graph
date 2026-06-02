"""Detect availability-zone redundancy requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("active_active_deployment", "high", (r"\bactive[-\s]?active\s+deployment\b", r"\bactive[-\s]?active\b")),
    ("availability_zone_redundancy", "high", (r"\bavailability\s+zone\s+redundancy\b", r"\baz\s+redundan(?:cy|t)\b")),
    ("multi_az", "high", (r"\bmulti[-\s]?az\b", r"\bmultiple\s+availability\s+zones\b")),
    ("region_failover", "high", (r"\bregion(?:al)?\s+failover\b", r"\bfailover\s+to\s+(?:another\s+)?region\b")),
    ("single_az_failure_tolerance", "high", (r"\bsingle[-\s]?az\s+failure\b", r"\bzone\s+failure\s+tolerance\b")),
)


def detect_query_availability_zone_redundancy_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, requirement_strength, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append(
                {
                    "matched_text": match.group(0),
                    "category": category,
                    "requirement_strength": requirement_strength,
                }
            )
    return sorted(rows, key=lambda row: row["category"])
