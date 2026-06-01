"""Detect sustainability requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("carbon_footprint", "high", (r"\bcarbon\s+footprint\b", r"\bcarbon\s+emissions?\b", r"\bco2e?\b")),
    ("energy_usage", "medium", (r"\benergy\s+(?:usage|use|consumption)\b", r"\bpower\s+consumption\b")),
    ("environmental_impact", "high", (r"\benvironmental\s+impact\b", r"\benvironmentally\s+friendly\b")),
    ("green_hosting", "medium", (r"\bgreen\s+hosting\b", r"\brenewable\s+(?:hosting|energy)\b")),
    ("resource_efficiency", "medium", (r"\bresource\s+efficiency\b", r"\befficient\s+resource\s+use\b")),
)


def detect_query_sustainability_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _CATEGORIES:
        matches = [m for pattern in patterns for m in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
