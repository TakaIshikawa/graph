"""Audit conditional coverage in RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text

_TYPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("prerequisite", re.compile(r"\b(?:if|only\s+when|requires?|provided\s+that)\b", re.I)),
    ("exception", re.compile(r"\b(?:unless|except|except\s+when|but\s+not)\b", re.I)),
    ("boundary", re.compile(r"\b(?:applies\s+(?:to|only\s+when)|limited\s+to|for\s+.+\s+only)\b", re.I)),
    ("fallback", re.compile(r"\b(?:otherwise|fallback|fall\s+back|if\s+not|alternatively)\b", re.I)),
)
_RECOMMEND_RE = re.compile(r"\b(?:should|must|recommend|use|choose)\b", re.I)


def audit_answer_condition_coverage(answer: str, evidence: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return condition cues and missing condition types."""
    text = " ".join(str(answer or "").split())
    answer_types = _matched_types(text)
    evidence_rows = []
    evidence_types: set[str] = set()
    for index, item in enumerate(evidence or []):
        snippet = content_text(item)
        types = _matched_types(snippet)
        if types:
            evidence_types.update(types)
            evidence_rows.append({"result_id": f"result-{index + 1}", "condition_types": sorted(types), "snippet": snippet[:160]})
    missing = sorted(evidence_types - answer_types)
    warnings = []
    if missing and _RECOMMEND_RE.search(text):
        warnings.append("unconditional_recommendation_with_conditional_evidence")
    return {
        "has_condition_cues": bool(answer_types),
        "answer_condition_types": sorted(answer_types),
        "missing_condition_types": missing,
        "matched_evidence_snippets": evidence_rows,
        "warnings": warnings,
    }


def _matched_types(text: str) -> set[str]:
    return {name for name, pattern in _TYPES if pattern.search(text)}
