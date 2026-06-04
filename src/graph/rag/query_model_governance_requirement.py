"""Detect model-governance requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTEXT_RE = re.compile(
    r"\b(?:ai|ml|machine\s+learning|model|llm|algorithmic|model\s+risk|ai\s+governance)\b",
    re.I,
)
_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("model_card", "medium", (r"\bmodel\s+cards?\b", r"\bmodel\s+documentation\b")),
    ("evaluation_metrics", "high", (r"\bevaluation\s+metrics?\b", r"\bmodel\s+metrics?\b", r"\bprecision\b", r"\brecall\b")),
    ("approval_workflow", "high", (r"\bapproval\s+workflows?\b", r"\bmodel\s+approval\b", r"\bgovernance\s+approval\b")),
    ("bias_testing", "high", (r"\bbias\s+tests?\b", r"\bbias\s+testing\b", r"\bfairness\s+tests?\b")),
    ("human_review", "high", (r"\bhuman\s+review\b", r"\bhuman[-\s]?in[-\s]?the[-\s]?loop\b", r"\bmanual\s+review\b")),
    ("monitoring_drift", "high", (r"\bdrift\s+monitoring\b", r"\bmonitor(?:ing)?\s+drift\b", r"\bmodel\s+drift\b")),
    ("training_data_lineage", "medium", (r"\btraining\s+data\s+lineage\b", r"\bdata\s+lineage\b", r"\btraining\s+data\s+provenance\b")),
    ("rollback_plan", "medium", (r"\brollback\s+plans?\b", r"\bmodel\s+rollback\b", r"\brollback\s+strategy\b")),
)


def detect_query_model_governance_requirement(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    if not _CONTEXT_RE.search(normalized):
        return {"has_model_governance_requirement": False, "requirements": [], "normalized_query": normalized}

    requirements = []
    for category, severity, patterns in _CATEGORY_SPECS:
        match = _first_match(normalized, patterns)
        if match:
            requirements.append({"category": category, "matched_text": match.group(0), "severity": severity})

    requirements.sort(key=lambda row: row["category"])
    return {
        "has_model_governance_requirement": bool(requirements),
        "requirements": requirements,
        "normalized_query": normalized,
    }


def _first_match(text: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
