"""Detect training material requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_TRAINING_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("onboarding", (r"\bonboarding\b",)),
    ("tutorial", (r"\btutorials?\b",)),
    ("training_deck", (r"\btraining\s+deck\b",)),
    ("playbook", (r"\bplaybook\b",)),
    ("workshop", (r"\bworkshop\b",)),
    ("certification", (r"\bcertification\b",)),
    ("enablement", (r"\benablement\b",)),
    ("internal_docs", (r"\binternal\s+docs?\b", r"\binternal\s+documentation\b")),
    ("hands_on_lab", (r"\bhands[-\s]on\s+lab\b",)),
)
_AUDIENCE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("employee", (r"\bemployees?\b",)),
    ("admin", (r"\badmins?\b", r"\badministrators?\b")),
    ("customer", (r"\bcustomers?\b",)),
    ("developer", (r"\bdevelopers?\b",)),
)
_ML_TRAINING = re.compile(r"\b(?:train|training)\s+(?:the\s+)?(?:model|classifier|embedding|llm)\b", re.I)


def detect_query_training_material_requirement(query: str) -> dict[str, Any]:
    """Return human training-material requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    training_terms = [] if _ML_TRAINING.search(text) else [term for term, patterns in _TRAINING_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    audience_terms = [term for term, patterns in _AUDIENCE_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)]
    recommendations = []
    if any(term in training_terms for term in {"internal_docs", "playbook", "tutorial"}):
        recommendations.append("prepare docs")
    if "hands_on_lab" in training_terms or "workshop" in training_terms:
        recommendations.append("prepare labs")
    if "enablement" in training_terms or "training_deck" in training_terms:
        recommendations.append("prepare enablement assets")
    return {
        "requires_training_material": bool(training_terms),
        "training_terms": training_terms,
        "audience_terms": audience_terms,
        "matched_phrases": training_terms,
        "recommendations": recommendations,
        "confidence": "high" if training_terms else "none",
    }
