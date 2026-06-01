"""Detect vendor risk requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_REQUIREMENT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vendor_risk", (r"\bvendor\s+risk\b", r"\bvendor\s+(?:security|compliance)\s+risk\b")),
    ("third_party_risk", (r"\bthird[-\s]party\s+risk\b", r"\bthird[-\s]party\s+(?:security|vendor)\s+risk\b")),
    ("supplier_assessment", (r"\bsupplier\s+assessment\b", r"\bsupplier\s+(?:security|risk)\s+assessment\b")),
    ("soc2_review", (r"\bsoc\s*2\s+review\b", r"\breview\s+(?:the\s+)?soc\s*2\b", r"\bsoc\s*2\s+report\s+review\b")),
    ("security_questionnaire", (r"\bsecurity\s+questionnaire\b", r"\bvendor\s+questionnaire\b")),
    ("subprocessors", (r"\bsubprocessors?\b", r"\bsub[-\s]processors?\b")),
    ("due_diligence", (r"\bdue\s+diligence\b", r"\bvendor\s+due\s+diligence\b", r"\bthird[-\s]party\s+due\s+diligence\b")),
)


def detect_query_vendor_risk_requirements(query: str) -> dict[str, Any]:
    """Return vendor risk requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    requirements = [
        requirement
        for requirement, patterns in _REQUIREMENT_SPECS
        if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    third_party_sensitive = any(requirement in {"third_party_risk", "subprocessors"} for requirement in requirements)
    return {
        "has_vendor_risk_requirements": bool(requirements),
        "requirements": requirements,
        "third_party_sensitive": third_party_sensitive,
    }
