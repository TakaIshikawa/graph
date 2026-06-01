"""Detect procurement approval requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_APPROVAL_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("procurement", (r"\bprocurement\s+approval\b", r"\bprocurement\s+process\b")),
    ("purchasing_approval", (r"\bpurchasing\s+approval\b", r"\bpurchase\s+approval\b")),
    ("vendor_review", (r"\bvendor\s+review\b", r"\bvendor\s+risk\s+review\b")),
    ("rfp", (r"\brfp\b", r"\brequest\s+for\s+proposal\b")),
    ("purchase_order", (r"\bpurchase\s+order\b", r"\bpo\s+required\b")),
    ("approved_vendor_list", (r"\bapproved\s+vendor\s+list\b", r"\bapproved\s+vendors?\b")),
)
_STAKEHOLDER_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("legal", (r"\blegal\s+review\b", r"\bcounsel\s+review\b")),
    ("finance", (r"\bfinance\s+approval\b", r"\bfinance\s+review\b")),
    ("security", (r"\bsecurity\s+review\b", r"\bsecurity\s+approval\b")),
)


def detect_query_procurement_approval_requirement(query: str) -> dict[str, Any]:
    """Return procurement approval signals mentioned by a query."""
    text = " ".join(str(query or "").split())
    approval_terms = [term for term, patterns in _APPROVAL_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    stakeholder_terms = [term for term, patterns in _STAKEHOLDER_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    matched_phrases = approval_terms + stakeholder_terms
    recommendations = []
    if approval_terms:
        recommendations.append("confirm procurement approval")
    if stakeholder_terms:
        recommendations.append("route stakeholder review")
    return {
        "requires_procurement_approval": bool(approval_terms),
        "approval_terms": approval_terms,
        "matched_phrases": matched_phrases,
        "stakeholder_terms": stakeholder_terms,
        "recommendations": recommendations,
        "confidence": "high" if approval_terms and stakeholder_terms else "medium" if approval_terms else "none",
    }
