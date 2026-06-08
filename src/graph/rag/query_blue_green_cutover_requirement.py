"""Detect blue-green cutover requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_DEPLOYMENT_CONTEXT = re.compile(
    r"\b(?:app|application|deploy(?:ment)?|environment|production|release|route|service|traffic)\b",
    re.I,
)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("blue_green_deployment", "high", (r"\bblue[-\s]?green\s+(?:cutover|deploy(?:ment)?|release|switch(?:over)?)\b",)),
    ("green_environment", "medium", (r"\bgreen\s+environment\b", r"\bgreen\s+stack\b", r"\bgreen\s+version\b")),
    ("traffic_switch", "high", (r"\btraffic\s+(?:switch|shift|flip|cutover)\b", r"\bswitch\s+traffic\b", r"\bflip\s+traffic\b")),
    ("route_cutover", "high", (r"\broute\s+cutover\b", r"\bdns\s+cutover\b", r"\bload\s+balancer\s+(?:flip|switch|cutover)\b")),
    ("old_environment_retention", "medium", (r"\bkeep\s+(?:the\s+)?old\s+environment\b", r"\bretain\s+(?:the\s+)?blue\s+environment\b")),
    ("instant_rollback", "high", (r"\binstant\s+rollback\b", r"\brollback\s+to\s+(?:blue|old)\b", r"\bquick\s+rollback\b")),
)


def detect_query_blue_green_cutover_requirement(query: str) -> dict[str, Any]:
    """Return blue-green cutover requirements mentioned by a query."""
    text = _normalize_query(query)
    requirements = []
    if _has_context(text):
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_blue_green_cutover_requirement": bool(requirements), "requirements": requirements}


def _has_context(text: str) -> bool:
    return bool(_DEPLOYMENT_CONTEXT.search(text) or re.search(r"\bblue[-\s]?green\b", text, re.I))


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
