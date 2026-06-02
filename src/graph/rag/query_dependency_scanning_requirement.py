"""Detect dependency scanning and SCA requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sca", (r"\bsca\b", r"\bsoftware\s+composition\s+analysis\b")),
    ("dependency_vulnerability_scanning", (r"\bdependency\s+vulnerability\s+scann?ing\b", r"\bscan\s+dependencies\b")),
    ("vulnerable_package_alerts", (r"\bvulnerable\s+package\s+alerts?\b", r"\bdependency\s+alerts?\b")),
    ("cvss_threshold", (r"\bcvss\s+threshold\b", r"\bcvss\s*(?:>=|>|at\s+least)\s*\d")),
    ("remediation_sla", (r"\bremediation\s+sla\b", r"\bremediate\s+within\b", r"\bfix\s+within\b")),
)
_SEVERITY_RE = re.compile(r"\bcvss\s*(?:>=|>|at\s+least)?\s*\d+(?:\.\d+)?\b|\b(?:critical|high|medium|low)\s+severity\b|\b\d+\s*(?:hours?|days?|weeks?)\b", re.I)


def detect_query_dependency_scanning_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {
        "requires_dependency_scanning": bool(categories),
        "cue_categories": categories,
        "severity_thresholds": _severity_thresholds(text) if categories else [],
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _severity_thresholds(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in _SEVERITY_RE.finditer(text):
        value = " ".join(match.group(0).split())
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
