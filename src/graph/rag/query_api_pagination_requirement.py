"""Detect API pagination requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_API_LIST_CONTEXT_RE = re.compile(
    r"\b(?:api|endpoint|rest|graphql|sdk|client|service|list(?:ing)?|list\s+(?:records?|items?|users?|orders?)|search|collection|feed|export|bulk\s+export|data\s+export|results?)\b",
    re.I,
)
_WEB_PAGE_ONLY_RE = re.compile(r"\b(?:web\s*page|webpage|page\s+layout|landing\s+page|ui|ux|website|html|css|carousel)\b", re.I)
_PAGINATION_CONTEXT_RE = re.compile(
    r"\b(?:paginat(?:e|ed|ion)|cursor|offset|limit|page\s+size|per[-\s]?page|next\s+(?:page\s+)?token|continuation\s+token|total\s+count|stable\s+ordering|sort\s+order)\b",
    re.I,
)
_REQUIREMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cursor_pagination", "high", (r"\bcursor[-\s]?based\s+pagination\b", r"\bcursor\s+pagination\b", r"\bpaginate\s+(?:with|by)\s+cursors?\b", r"\bcursors?\s+(?:for|in)\s+(?:api\s+)?pagination\b")),
    ("next_token", "high", (r"\bnext\s+(?:page\s+)?tokens?\b", r"\bcontinuation\s+tokens?\b", r"\bpage\s+tokens?\b", r"\bnext_token\b")),
    ("offset_pagination", "medium", (r"\boffset[-\s]?based\s+pagination\b", r"\boffset\s+pagination\b", r"\blimit\s*/\s*offset\b", r"\boffset\s+and\s+limit\b")),
    ("page_size_limit", "medium", (r"\bpage\s+size\s+limits?\b", r"\bmax(?:imum)?\s+page\s+size\b", r"\bper[-\s]?page\s+limits?\b", r"\blimit\s+(?:parameter|params?|value|cap|maximum)\b")),
    ("stable_ordering", "high", (r"\bstable\s+ordering\b", r"\bconsistent\s+sort(?:ing)?\b", r"\bdeterministic\s+order(?:ing)?\b", r"\bsort\s+order\s+(?:for|with)\s+pagination\b")),
    ("total_count", "medium", (r"\btotal\s+counts?\b", r"\btotal_count\b", r"\binclude\s+totals?\b", r"\bcount\s+(?:of|for)\s+all\s+(?:records?|items?|results?)\b")),
)


def detect_query_api_pagination_requirements(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    has_context = bool(_API_LIST_CONTEXT_RE.search(text) and _PAGINATION_CONTEXT_RE.search(text))
    if _WEB_PAGE_ONLY_RE.search(text) and not re.search(r"\b(?:api|endpoint|rest|graphql|sdk|client|service|export|list(?:ing)?|results?)\b", text, re.I):
        has_context = False

    requirements = []
    if has_context:
        for category, severity, patterns in _REQUIREMENTS:
            match = _first_match(patterns, text)
            if match:
                requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})
    requirements.sort(key=lambda row: row["category"])
    return {"has_api_pagination_requirements": bool(requirements), "requirements": requirements}


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
