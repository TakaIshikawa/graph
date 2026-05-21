"""Classify query stakes for RAG planning."""

from __future__ import annotations

import re
from typing import Any

_DOMAINS = {
    "medical": ("diagnosis", "symptom", "medicine", "drug", "dosage", "doctor", "cancer", "treatment"),
    "legal": ("law", "legal", "lawsuit", "contract", "tenant", "court", "attorney", "compliance"),
    "financial": ("invest", "loan", "tax", "retirement", "mortgage", "stock", "bankruptcy", "insurance"),
    "safety": ("danger", "hazard", "emergency", "fire", "electrical", "poison", "evacuate", "unsafe"),
    "employment": ("hire", "fire", "layoff", "salary", "workplace", "employee", "interview", "severance"),
    "privacy": ("ssn", "social security", "password", "personal data", "pii", "email address", "phone number"),
}


def classify_query_stakes(query: str) -> dict[str, Any]:
    """Return a deterministic low/medium/high stakes classification."""
    text = " ".join(str(query or "").split())
    lowered = text.casefold()
    matched = [
        {"domain": domain, "matched_cues": [cue for cue in cues if _contains(lowered, cue)]}
        for domain, cues in _DOMAINS.items()
    ]
    matched = [row for row in matched if row["matched_cues"]]
    high_domains = {"medical", "legal", "financial", "safety", "privacy"}
    if any(row["domain"] in high_domains for row in matched):
        level = "high"
    elif matched:
        level = "medium"
    else:
        level = "low"
    safeguards = []
    if level in {"medium", "high"}:
        safeguards.extend(["require_citations", "include_uncertainty"])
    if level == "high":
        safeguards.extend(["prefer_primary_sources", "avoid_overconfident_advice"])
    return {"stakes": level, "domains": matched, "safeguards": safeguards}


def _contains(text: str, cue: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text))
