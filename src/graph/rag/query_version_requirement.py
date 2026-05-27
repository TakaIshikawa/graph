"""Detect version and compatibility requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SEMVER_RE = re.compile(r"\b(?:v(?:ersion)?\s*)?(\d+\.\d+(?:\.\d+)?)(?:\s*(?:x|lts))?\b", re.I)
_V_MAJOR_RE = re.compile(r"\bv\s*(\d+)\b|\bversion\s+(\d+)\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2})\s*(?:edition|release|version|api|standard|spec(?:ification)?)\b", re.I)
_EDITION_RE = re.compile(r"\b(?:edition|release)\s+(20\d{2})\b", re.I)
_NAMED_VERSION_RE = re.compile(
    r"\b(?:python|node(?:\.js)?|java|go|ruby|rails|django|react|vue|angular|api|sdk)\s+"
    r"(?:v(?:ersion)?\s*)?(\d+(?:\.\d+){0,2})\b",
    re.I,
)
_FRESHNESS_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("latest", re.compile(r"\blatest\b", re.I)),
    ("current", re.compile(r"\bcurrent\b", re.I)),
    ("newest", re.compile(r"\bnewest\b", re.I)),
    ("latest_lts", re.compile(r"\blatest\s+lts\b|\blts\s+release\b|\blts\s+version\b", re.I)),
)
_COMPATIBILITY_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("backward_compatible", re.compile(r"\bbackwards?\s+compatible\b|\bbackward compatibility\b", re.I)),
    ("forward_compatible", re.compile(r"\bforwards?\s+compatible\b|\bforward compatibility\b", re.I)),
    ("compatibility_target", re.compile(r"\bcompatible\s+with\b|\bworks?\s+with\b|\bsupports?\b", re.I)),
    ("migration", re.compile(r"\bmigrat(?:e|ing|ion)\b|\bupgrade(?:s|d| path)?\b", re.I)),
)


def detect_query_version_requirement(query: str) -> dict[str, Any]:
    """Return concrete version targets and compatibility cues in a query."""
    normalized = _inline_text(query)
    versions = _extract_versions(normalized)
    freshness_cues = [label for label, pattern in _FRESHNESS_CUES if pattern.search(normalized)]
    compatibility_cues = [label for label, pattern in _COMPATIBILITY_CUES if pattern.search(normalized)]
    has_requirement = bool(versions or freshness_cues or compatibility_cues)
    return {
        "has_version_requirement": has_requirement,
        "versions": versions,
        "compatibility_cues": compatibility_cues,
        "freshness_sensitive": bool(freshness_cues),
    }


def _extract_versions(query: str) -> list[str]:
    versions: list[str] = []
    seen: set[str] = set()

    def add(version: str) -> None:
        normalized = version.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            versions.append(normalized)

    for pattern in (_NAMED_VERSION_RE, _SEMVER_RE):
        for match in pattern.finditer(query):
            add(match.group(1))

    for match in _V_MAJOR_RE.finditer(query):
        add(next(group for group in match.groups() if group))

    for pattern in (_YEAR_RE, _EDITION_RE):
        for match in pattern.finditer(query):
            add(match.group(1))

    return versions


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
