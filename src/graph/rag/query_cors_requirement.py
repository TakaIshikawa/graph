"""Detect CORS requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CORS_CONTEXT_RE = re.compile(
    r"\b(?:cors|cross[-\s]?origin|browser|frontend|front[-\s]?end|web\s+app|api|endpoint|xmlhttprequest|fetch\s+requests?|preflight)\b",
    re.I,
)
_CORS_SIGNAL_RE = re.compile(
    r"\b(?:cors|cross[-\s]?origin|access-control-|preflight|allowed\s+(?:origins?|methods?|headers?)|exposed\s+headers?|credentials?|max[-\s]?age)\b",
    re.I,
)
_GENERIC_ORIGIN_RE = re.compile(r"\b(?:origin|origins|source|sources)\b", re.I)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("allowed_headers", "medium", (r"\baccess-control-allow-headers\b", r"\ballowed\s+headers?\b", r"\ballow\s+headers?\b")),
    ("allowed_methods", "medium", (r"\baccess-control-allow-methods\b", r"\ballowed\s+methods?\b", r"\ballow\s+(?:get|post|put|patch|delete|options)(?:\s*,?\s*(?:get|post|put|patch|delete|options))*\b")),
    ("allowed_origins", "high", (r"\baccess-control-allow-origin\b", r"\ballowed\s+origins?\b", r"\ballowlist\s+origins?\b", r"\borigin\s+allowlists?\b")),
    ("credentials", "high", (r"\baccess-control-allow-credentials\b", r"\bcredentialed\s+(?:cors\s+)?requests?\b", r"\bwith\s+credentials\b", r"\ballow\s+credentials\b")),
    ("exposed_headers", "medium", (r"\baccess-control-expose-headers\b", r"\bexposed\s+headers?\b", r"\bexpose\s+headers?\b")),
    ("max_age", "low", (r"\baccess-control-max-age\b", r"\bpreflight\s+max[-\s]?age\b", r"\bmax[-\s]?age\s+for\s+preflight\b")),
    ("preflight", "high", (r"\bpreflight\s+(?:requests?|handling|checks?|responses?)\b", r"\boptions\s+preflight\b", r"\bhandle\s+preflight\b")),
)


def detect_query_cors_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_context = bool(_CORS_CONTEXT_RE.search(text) and _CORS_SIGNAL_RE.search(text))
    if _GENERIC_ORIGIN_RE.search(text) and not re.search(r"\b(?:cors|cross[-\s]?origin|browser|frontend|front[-\s]?end|web\s+app|api|endpoint|preflight|access-control-)\b", text, re.I):
        has_context = False

    requirements = []
    if has_context:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_cors_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
