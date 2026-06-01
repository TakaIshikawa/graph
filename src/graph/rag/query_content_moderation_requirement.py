"""Detect content moderation requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_POLICY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("moderation", "medium", (r"\bcontent\s+moderation\b", r"\bmoderation\s+policy\b")),
    ("abuse", "medium", (r"\babuse\s+review\b", r"\babusive\s+content\b")),
    ("hate", "high", (r"\bhate\s+(?:speech|content)\b", r"\bhateful\s+content\b")),
    ("harassment", "high", (r"\bharassment\b", r"\bharassing\s+content\b")),
    ("spam", "medium", (r"\bspam\b", r"\bspam\s+(?:detection|filtering|prevention)\b", r"\bfilter\s+spam\b")),
    ("csam", "high", (r"\bcsam\b", r"\bchild\s+sexual\s+abuse\s+material\b")),
    ("ugc_safety", "medium", (r"\buser[-\s]generated\s+content\s+safety\b", r"\bugc\s+safety\b")),
)
_HUMAN_REVIEW_PATTERNS = (r"\bhuman\s+review\b", r"\bmoderator\s+review\b", r"\bmanual\s+queue\b", r"\bescalat(?:e|ion)\s+to\s+(?:a\s+)?moderator\b")


def detect_query_content_moderation_requirement(query: str) -> dict[str, Any]:
    """Return content moderation requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    policy_areas = [
        area for area, _severity, patterns in _POLICY_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    human_review_required = any(re.search(pattern, text, re.I) for pattern in _HUMAN_REVIEW_PATTERNS)
    high_areas = {area for area, severity, _patterns in _POLICY_SPECS if severity == "high"}
    severity = "high" if any(area in high_areas for area in policy_areas) else "medium" if policy_areas else "none"
    return {
        "requires_content_moderation": bool(policy_areas),
        "policy_areas": policy_areas,
        "human_review_required": human_review_required,
        "matched_cues": policy_areas,
        "severity": severity,
    }
