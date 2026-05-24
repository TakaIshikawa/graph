"""Audit answers for ranking and percentile claims unsupported by evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_RANK_RE = re.compile(r"\b(?:top\s+\d+%|bottom\s+\d+%|\d+(?:st|nd|rd|th)\s+percentile|percentile|ranked?\s+#?\d+|highest|lowest|best|worst|quartile|decile)(?=\W|$)", re.I)


def audit_answer_percentile_rankings(answer: str, evidence: Iterable[Any]) -> dict[str, Any]:
    """Flag ranking claims when no ranking-compatible evidence is visible."""
    answer_text = " ".join(str(answer or "").split())
    support_rows = _support_rows(evidence)
    findings = []
    for sentence in _sentences(answer_text):
        terms = _ranking_terms(sentence)
        if not terms:
            continue
        supporting_ids = [row["evidence_id"] for row in support_rows]
        if supporting_ids:
            continue
        findings.append(
            {
                "claim_text": sentence,
                "ranking_terms": terms,
                "supporting_evidence_ids": [],
                "severity": "medium",
                "reason_codes": ["ranking_claim_without_ranking_evidence"],
            }
        )
    return {"findings": findings, "ranking_evidence_count": len(support_rows)}


def _support_rows(evidence: Iterable[Any]) -> list[dict[str, str]]:
    rows = []
    for index, record in enumerate(evidence or []):
        if _RANK_RE.search(content_text(record)):
            rows.append({"evidence_id": result_id(record, index)})
    return rows


def _ranking_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for match in _RANK_RE.finditer(text):
        term = match.group(0).casefold()
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
