"""Detect FedRAMP requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("fedramp", "high", (r"\bfedramp\b", r"\bfederal\s+risk\s+and\s+authorization\s+management\s+program\b")),
    ("ato", "high", (r"\bagency\s+ato\b", r"\bauthority\s+to\s+operate\b")),
    ("jab", "medium", (r"\bjab\s+authorization\b", r"\bjoint\s+authorization\s+board\b")),
    ("baseline", "medium", (r"\bfedramp\s+(?:moderate|high)\b", r"\b(?:moderate|high)\s+baselines?\b")),
    ("3pao", "medium", (r"\b3pao\b", r"\bthird[- ]party\s+assessment\s+organization\b")),
    ("poam", "medium", (r"\bpoa&m\b", r"\bplan\s+of\s+action\s+and\s+milestones\b")),
    ("continuous_monitoring", "medium", (r"\bcontinuous\s+monitoring\b", r"\bconmon\b")),
)


def detect_query_fedramp_requirement(query: str) -> dict[str, Any]:
    matches = _matches(query)
    categories = sorted(dict.fromkeys(match["category"] for match in matches))
    return {"requires_fedramp": bool(matches), "categories": categories, "matches": matches}


def _matches(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": match.span()})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"], row["matched_text"].casefold()))
