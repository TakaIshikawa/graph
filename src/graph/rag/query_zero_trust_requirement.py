"""Detect zero-trust requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:zero[-\s]?trust|never\s+trust\s+always\s+verify|continuous\s+verification|"
    r"least\s+privilege\s+access|explicit\s+verification|identity[-\s]?aware\s+access|microsegmentation)\b",
    re.I,
)

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("continuous_evaluation", "high", (r"\bcontinuous\s+(?:verification|evaluation)\b", r"\bcontinuous\s+access\s+evaluation\b")),
    ("device_context", "medium", (r"\bdevice\s+context\b", r"\bdevice\s+posture\b", r"\bendpoint\s+context\b")),
    ("identity_verification", "high", (r"\bexplicit\s+verification\b", r"\bidentity[-\s]?aware\s+access\b", r"\bnever\s+trust\s+always\s+verify\b")),
    ("least_privilege", "high", (r"\bleast\s+privilege(?:\s+access)?\b", r"\bjust[-\s]?enough\s+access\b")),
    ("network_segmentation", "high", (r"\bmicrosegmentation\b", r"\bmicro[-\s]?segmentation\b", r"\bnetwork\s+segmentation\b")),
    ("policy_enforcement", "medium", (r"\bpolicy\s+enforcement\b", r"\bzero[-\s]?trust\s+polic(?:y|ies)\b", r"\bcontextual\s+access\s+polic(?:y|ies)\b")),
)


def detect_query_zero_trust_requirements(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_zero_trust_requirements": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_zero_trust_requirements": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
