"""Audit temporal specificity in RAG answers."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_TEMPORAL_RE = re.compile(r"\b(?:recently|currently|latest|soon|now|today|this year|last year|new|updated|trend|trends)\b", re.I)
_SPECIFIC_DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?\b|\b\d{4}-\d{2}-\d{2}\b")


def audit_answer_temporal_specificity(answer: Any) -> dict[str, Any]:
    """Return counts for temporal claims with explicit versus vague timing."""
    claims = [sentence for sentence in _sentences(answer) if _TEMPORAL_RE.search(sentence) or _SPECIFIC_DATE_RE.search(sentence)]
    dated = [sentence for sentence in claims if _SPECIFIC_DATE_RE.search(sentence)]
    vague = [sentence for sentence in claims if _TEMPORAL_RE.search(sentence) and not _SPECIFIC_DATE_RE.search(sentence)]
    findings = [{"type": "vague_temporal_claim", "snippet": sentence} for sentence in vague]

    return {
        "temporal_claim_count": len(claims),
        "dated_claim_count": len(dated),
        "vague_temporal_claim_count": len(vague),
        "specificity_ratio": 0.0 if not claims else round(len(dated) / len(claims), 4),
        "findings": findings,
    }


def _sentences(answer: Any) -> list[str]:
    return [" ".join(match.group(0).strip().split()) for match in _SENTENCE_RE.finditer(string(answer) or "") if match.group(0).strip()]
