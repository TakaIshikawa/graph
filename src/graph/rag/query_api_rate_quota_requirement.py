"""Detect API rate quota requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("api_quota", (r"\bapi\s+quota\b", r"\brate\s+quota\b")),
    ("request_limit", (r"\brequest\s+limits?\b", r"\brate\s+limits?\b", r"\blimit\s+requests?\b")),
    ("monthly_call_allowance", (r"\bmonthly\s+(?:call|request)\s+allowance\b", r"\b(?:calls|requests)\s+per\s+month\b")),
    ("burst_limit", (r"\bburst\s+(?:limit|quota)\b", r"\bburst\s+capacity\b")),
    ("quota_increase", (r"\bquota\s+increase\b", r"\bincrease\s+(?:the\s+)?quota\b", r"\brequest\s+higher\s+quota\b")),
)
_NUMERIC_QUOTA_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:api\s+)?(?:calls|requests|reqs)\s+per\s+(?:second|minute|hour|day|month)\b",
    re.I,
)


def detect_query_api_rate_quota_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    cue_categories = _matched_categories(text)
    numeric_quotas = [match.group(0) for match in _NUMERIC_QUOTA_RE.finditer(text)]
    return {
        "requires_api_rate_quota": bool(cue_categories or numeric_quotas),
        "cue_categories": cue_categories,
        "numeric_quotas": numeric_quotas,
    }


def _matched_categories(text: str) -> list[str]:
    return [category for category, patterns in _CUES if any(re.search(pattern, text, re.I) for pattern in patterns)]


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.split())
