"""Detect browser support evidence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("browser_family", (r"\b(?:chrome|safari|firefox|edge|chromium|webkit|ie\s*11|internet\s+explorer)\b",)),
    ("version_floor", (r"\b(?:minimum|min|at least|version floor|support(?:s|ed)? since)\s+(?:chrome|safari|firefox|edge|browser)?\s*\d+", r"\b(?:chrome|safari|firefox|edge)\s*(?:>=|>|at least|version\s+)?\s*\d+")),
    ("legacy_browser", (r"\blegacy\s+browsers?\b", r"\bie\s*11\b", r"\binternet\s+explorer\b", r"\bfirefox\s+esr\b")),
    ("rendering_engine", (r"\b(?:chromium|webkit|blink|gecko)\b",)),
    ("compatibility_matrix", (r"\bcompatibility\s+matrix\b", r"\bbrowser\s+(?:support|compatibility)\b", r"\bsupported\s+browsers?\b")),
)

_VALUE_PATTERNS: tuple[str, ...] = (
    r"\bChrome\s+\d+(?:\.\d+)?\b",
    r"\bSafari\s+\d+(?:\.\d+)?\b",
    r"\bFirefox\s+ESR\b",
    r"\bFirefox\s+\d+(?:\.\d+)?\b",
    r"\bEdge\s+\d+(?:\.\d+)?\b",
    r"\bIE\s*11\b",
    r"\bInternet\s+Explorer\s+11\b",
    r"\bChromium\b",
    r"\bWebKit\b",
    r"\bBlink\b",
    r"\bGecko\b",
)


def detect_query_browser_support_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    cue_categories = _matched_categories(text)
    browser_versions = _extract_values(text)
    return {
        "requires_browser_support": bool(cue_categories or browser_versions),
        "cue_categories": cue_categories,
        "browser_versions": browser_versions,
    }


def _matched_categories(text: str) -> list[str]:
    return [category for category, patterns in _CUES if any(re.search(pattern, text, re.I) for pattern in patterns)]


def _extract_values(text: str) -> list[str]:
    values: list[tuple[int, str]] = []
    for pattern in _VALUE_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            values.append((match.start(), " ".join(match.group(0).split())))
    return list(dict.fromkeys(value for _pos, value in sorted(values)))


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.split())
