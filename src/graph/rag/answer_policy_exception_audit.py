"""Audit answers for policy claims that omit available exception evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_RULE_RE = re.compile(r"\b(?:must|required|prohibited|allowed|eligible|compliant|mandatory|shall|may not)\b", re.I)
_EXCEPTION_RE = re.compile(r"\b(?:except|exception|exemption|waiver|unless|grandfathered|case-by-case|discretionary)\b", re.I)


def audit_answer_policy_exceptions(answer: str, evidence: Iterable[Any]) -> dict[str, Any]:
    """Flag rule-like answer sentences that omit exception evidence."""
    answer_text = " ".join(str(answer or "").split())
    exception_rows = _exception_rows(evidence)
    findings = []
    if exception_rows and not _EXCEPTION_RE.search(answer_text):
        for sentence in _sentences(answer_text):
            if not _RULE_RE.search(sentence):
                continue
            findings.append(
                {
                    "claim_text": sentence,
                    "exception_evidence_ids": [row["evidence_id"] for row in exception_rows],
                    "severity": "medium",
                    "reason_codes": ["rule_claim_omits_available_exception"],
                }
            )
    return {"findings": findings, "exception_evidence_count": len(exception_rows)}


def _exception_rows(evidence: Iterable[Any]) -> list[dict[str, str]]:
    rows = []
    for index, record in enumerate(evidence or []):
        if _EXCEPTION_RE.search(content_text(record)):
            rows.append({"evidence_id": result_id(record, index)})
    return rows


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
