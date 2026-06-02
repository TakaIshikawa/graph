"""Detect software and content license-compliance requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("license_compliance", (r"\blicen[cs]e\s+compliance\b", r"\blicen[cs]e\s+obligations?\b")),
    ("oss_license", (r"\boss\s+licen[cs]e\b", r"\bopen\s+source\s+licen[cs]e\b")),
    ("copyleft", (r"\bcopyleft\b", r"\bstrong\s+copyleft\b")),
    ("commercial_use", (r"\bcommercial\s+use\b", r"\bcommercial\s+licen[cs]e\b")),
    ("attribution", (r"\battribution\s+requirements?\b", r"\brequire(?:d)?\s+attribution\b")),
    ("redistribution", (r"\bredistribution\b", r"\bredistribute\b")),
    ("spdx", (r"\bspdx\b", r"\bspdx\s+identifier\b")),
    ("dependency_license_review", (r"\bdependency\s+licen[cs]e\s+review\b", r"\blicen[cs]e\s+review\s+for\s+dependencies\b")),
)
_LICENSES: tuple[tuple[str, str], ...] = (
    ("MIT", r"\bmit\b"),
    ("Apache-2.0", r"\bapache[-\s]?2(?:\.0)?\b"),
    ("AGPL", r"\bagpl(?:[-\s]?3(?:\.0)?)?\b"),
    ("LGPL", r"\blgpl(?:[-\s]?3(?:\.0)?)?\b"),
    ("GPL", r"\bgpl(?:[-\s]?3(?:\.0)?)?\b"),
    ("BSD", r"\bbsd(?:[-\s]?\d[-\s]?clause)?\b"),
    ("MPL", r"\bmpl(?:[-\s]?2(?:\.0)?)?\b"),
    ("Creative Commons", r"\bcreative\s+commons\b|\bcc[-\s]by(?:[-\s]sa|[-\s]nc)?\b"),
)


def detect_query_license_compliance_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    categories = [category for category, patterns in _CATEGORIES if _first_match(patterns, text)]
    return {
        "requires_license_compliance": bool(categories),
        "cue_categories": categories,
        "license_names": _license_names(text) if categories else [],
    }


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _license_names(text: str) -> list[str]:
    return [name for name, pattern in _LICENSES if re.search(pattern, text, re.I)]


def _normalize_query(query: str) -> str:
    text = " ".join(str(query or "").split())
    if not text:
        raise ValueError("query must not be empty")
    return text
