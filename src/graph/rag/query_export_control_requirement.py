"""Detect export-control requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_FRAMEWORKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ear", (r"\bear\b", r"\bexport\s+administration\s+regulations?\b")),
    ("itar", (r"\bitar\b", r"\binternational\s+traffic\s+in\s+arms\s+regulations?\b")),
    ("ofac", (r"\bofac\b", r"\boffice\s+of\s+foreign\s+assets\s+control\b")),
)
_CUES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("embargo", "high", (r"\bembargo(?:ed)?\s+(?:countries|country|regions?|markets?)\b", r"\bsanction(?:ed)?\s+(?:countries|country|regions?|markets?)\b")),
    ("denied_party", "high", (r"\bdenied[-\s]?part(?:y|ies)\s+screening\b", r"\brestricted[-\s]?part(?:y|ies)\s+screening\b", r"\bsanctions?\s+screening\b")),
    ("dual_use", "medium", (r"\bdual[-\s]?use\b", r"\bexport[-\s]?controlled\s+(?:technology|data|items?|software)\b")),
    ("export_control", "high", (r"\bexport\s+controls?\b", r"\bexport[-\s]?control(?:led)?\s+review\b")),
    ("sanctions", "high", (r"\bsanctions?\s+(?:compliance|review|requirements?)\b", r"\btrade\s+sanctions?\b")),
)


def detect_query_export_control_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    frameworks = [name for name, patterns in _FRAMEWORKS if _matches_any(patterns, text)]
    matched_cues = []
    severities = []
    for category, severity, patterns in _CUES:
        match = _first_match(patterns, text)
        if match:
            matched_cues.append({"category": category, "matched_text": match.group(0)})
            severities.append(severity)

    requires_review = bool(frameworks or matched_cues)
    return {
        "requires_export_control_review": requires_review,
        "frameworks": frameworks,
        "matched_cues": matched_cues,
        "severity": _highest_severity(severities or (["high"] if frameworks else [])),
    }


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _highest_severity(severities: list[str]) -> str:
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "none"


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
