"""Audit answers for unacknowledged counterexample evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, tokens

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_COUNTER_RE = re.compile(
    r"\b(?:however|except|but|contrary|in contrast|although|whereas|limitation|limited|caveat|unless|failed to|did not|does not|no evidence)\b",
    re.I,
)
_ACK_RE = re.compile(r"\b(?:however|except|but|although|caveat|limitation|counterexample|mixed|not all|some|may|might)\b", re.I)
_BROAD_RE = re.compile(r"\b(?:all|always|never|every|none|proves?|shows?|demonstrates?|clearly|definitively|will|must)\b", re.I)


def audit_answer_missing_counterexamples(answer: str, evidence: Iterable[Any]) -> dict[str, Any]:
    """Flag broad answer claims when counterexample-like evidence is omitted."""
    answer_text = " ".join(str(answer or "").split())
    evidence_rows = _counterexample_rows(evidence)
    findings: list[dict[str, Any]] = []
    if not answer_text or not evidence_rows:
        return {"findings": [], "counterexample_evidence_count": len(evidence_rows)}

    acknowledged = bool(_ACK_RE.search(answer_text))
    for sentence in _sentences(answer_text):
        if acknowledged or not _BROAD_RE.search(sentence):
            continue
        sentence_terms = tokens(sentence, min_length=4)
        matches = [
            row
            for row in evidence_rows
            if not sentence_terms or sentence_terms & row["terms"]
        ]
        if not matches:
            continue
        severity = "high" if len(matches) > 1 else "medium"
        findings.append(
            {
                "claim_text": sentence,
                "counterexample_evidence_ids": [row["evidence_id"] for row in matches],
                "severity": severity,
                "reason_codes": ["broad_claim", "unacknowledged_counterexample"],
            }
        )
    return {"findings": findings, "counterexample_evidence_count": len(evidence_rows)}


def _counterexample_rows(evidence: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(evidence):
        text = content_text(record)
        if _COUNTER_RE.search(text):
            rows.append({"evidence_id": result_id(record, index), "text": text, "terms": tokens(text, min_length=4)})
    return rows


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
