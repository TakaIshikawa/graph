"""Detect API deprecation policy requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_PATTERN = re.compile(
    r"\b(?:api|apis|endpoint|endpoints|sdk|platform|service|services|version|versions|v\d+|client|clients|integration|integrations)\b",
    re.I,
)
_DEPRECATION_PATTERN = re.compile(r"\b(?:deprecat(?:e|ed|ing|ion)|sunset|retire(?:d|s|ment)?|end[-\s]?of[-\s]?life|eol)\b", re.I)

_CATEGORY_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "notice_period",
        "high",
        re.compile(r"\b(?:notice\s+period|advance\s+notice|deprecation\s+notice|notify\s+(?:users|customers|clients)|notification\s+timeline)\b", re.I),
    ),
    (
        "sunset_date",
        "high",
        re.compile(r"\b(?:sunset\s+date|sunset\s+timeline|sunset\s+schedule|retirement\s+date|removal\s+date|end[-\s]?of[-\s]?life\s+date|eol\s+date)\b", re.I),
    ),
    (
        "migration_guide",
        "medium",
        re.compile(r"\b(?:migration\s+guide|migration\s+path|upgrade\s+guide|replacement\s+api|replacement\s+endpoint|transition\s+guide)\b", re.I),
    ),
    (
        "version_support",
        "medium",
        re.compile(r"\b(?:version\s+support\s+window|support\s+window|supported\s+versions?|version\s+lifecycle|how\s+long\s+.*\bversions?\b)\b", re.I),
    ),
    (
        "backward_compatibility",
        "medium",
        re.compile(r"\b(?:backward[-\s]compatibility|backwards[-\s]compatibility|backward[-\s]compatible|backwards[-\s]compatible|breaking\s+changes?|legacy\s+(?:clients?|versions?|support))\b", re.I),
    ),
)


def detect_query_deprecation_policy_requirements(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    gated = _has_policy_context(normalized)
    matches = _matches(normalized) if gated else []
    return {
        "has_deprecation_policy_requirements": bool(matches),
        "requirements": matches,
        "normalized_query": normalized,
    }


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())


def _has_policy_context(normalized_query: str) -> bool:
    return bool(_DEPRECATION_PATTERN.search(normalized_query) and _CONTEXT_PATTERN.search(normalized_query))


def _matches(normalized_query: str) -> list[dict[str, Any]]:
    rows = []
    for category, severity, pattern in _CATEGORY_SPECS:
        match = pattern.search(normalized_query)
        if match:
            rows.append({"category": category, "severity": severity, "matched_text": match.group(0), "span": [match.start(), match.end()]})
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["category"]))
    return rows
