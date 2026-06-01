"""Detect privacy impact requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dpia", (r"\bdpia\b", r"\bdata\s+protection\s+impact\s+assessment\b")),
    ("pia", (r"\bpia\b", r"\bprivacy\s+impact\s+assessment\b")),
    ("privacy_impact_assessment", (r"\bprivacy\s+impact\s+assessment\b", r"\bprivacy\s+impact\s+review\b")),
    ("data_protection_impact", (r"\bdata\s+protection\s+impact\b", r"\bdata\s+protection\s+impact\s+assessment\b")),
    ("privacy_review", (r"\bprivacy\s+review\b", r"\breview\s+privacy\s+impact\b")),
    ("high_risk_processing", (r"\bhigh[-\s]risk\s+processing\b", r"\bhigh\s+risk\s+data\s+processing\b")),
)


def detect_query_privacy_impact_requirements(query: str) -> dict[str, Any]:
    """Return privacy impact requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    requirements = [
        requirement
        for requirement, patterns in _REQUIREMENT_SPECS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    return {
        "has_privacy_impact_requirements": bool(requirements),
        "requirements": requirements,
        "high_risk_processing_sensitive": "high_risk_processing" in requirements,
    }
