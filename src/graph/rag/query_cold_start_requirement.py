"""Detect cold-start requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cold_start_latency",
        (
            r"\bcold[-\s]?start\s+(?:latency|delay|time|performance)\b",
            r"\blatency\s+from\s+cold[-\s]?starts?\b",
            r"\bcold[-\s]?starts?\b",
        ),
    ),
    (
        "warm_pool",
        (
            r"\bwarm\s+pools?\b",
            r"\bkeep\s+(?:instances?|workers?|containers?)\s+warm\b",
            r"\bwarmed\s+(?:instances?|workers?|containers?)\b",
        ),
    ),
    (
        "prewarming",
        (
            r"\bpre[-\s]?warm(?:ing|s|ed)?\b",
            r"\bwarm[-\s]?up\s+(?:requests?|probes?|jobs?)\b",
            r"\bstartup\s+warming\b",
        ),
    ),
    (
        "provisioned_concurrency",
        (
            r"\bprovisioned\s+concurrency\b",
            r"\breserved\s+concurrency\b",
            r"\bminimum\s+instances?\b",
            r"\bmin\s+instances?\b",
        ),
    ),
    (
        "first_request_penalty",
        (
            r"\bfirst[-\s]?request\s+(?:penalty|latency|delay)\b",
            r"\bfirst\s+request\s+is\s+slow\b",
            r"\binitial\s+request\s+(?:penalty|latency|delay)\b",
        ),
    ),
    (
        "scale_from_zero",
        (
            r"\bscale[-\s]?from[-\s]?zero\b",
            r"\bscaling\s+from\s+zero\b",
            r"\bzero\s+to\s+(?:one|many|N)\b",
            r"\bserverless\s+scale\s+up\b",
        ),
    ),
)

_ORDINARY_STARTUP_PATTERNS = (
    r"\bstartup\s+(?:company|launch|announcement|program|costs?)\b",
    r"\bproduct\s+launch\b",
    r"\blaunch\s+plan\b",
)


def detect_query_cold_start_requirement(query: str) -> dict[str, Any]:
    """Return cold-start and prewarming requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    rows = [] if _ordinary_startup_only(text) else _requirement_rows(text)
    categories = [row["category"] for row in rows]
    has_cold_start_requirement = bool(rows)

    return {
        "has_cold_start_requirement": has_cold_start_requirement,
        "requirements": rows,
        "categories": categories,
        "matched_phrases": [row["matched_text"] for row in rows],
        "confidence": _confidence(categories),
    }


def _requirement_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, patterns in _CATEGORY_SPECS:
        match = _first_match(patterns, text)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "span": (match.start(), match.end())})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"]))


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _ordinary_startup_only(text: str) -> bool:
    if not text:
        return False
    has_startup = any(re.search(pattern, text, re.I) for pattern in _ORDINARY_STARTUP_PATTERNS)
    has_cold_start = re.search(r"\bcold[-\s]?starts?\b|\bpre[-\s]?warm|\bwarm\s+pools?\b", text, re.I)
    return bool(has_startup and not has_cold_start)


def _confidence(categories: list[str]) -> str:
    if not categories:
        return "none"
    if len(categories) >= 2:
        return "high"
    return "medium"
