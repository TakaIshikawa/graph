"""Compare quoted evidence snippets with cited answer claims."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, tokens, value

_NUMBER_RE = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?%?")
_NEG_RE = re.compile(r"\b(?:no|not|never|without|failed|lack|lacks|cannot|didn't|doesn't)\b", re.I)


def audit_evidence_quoted_claim_alignment(claims: Iterable[Any], quoted_evidence: Iterable[Any]) -> dict[str, Any]:
    """Score lexical, numeric, and polarity agreement for claim-evidence pairs."""
    evidence_by_id = {_record_id(row, index): row for index, row in enumerate(quoted_evidence)}
    rows = []
    for index, claim in enumerate(claims):
        claim_text = _record_text(claim)
        evidence_id = _evidence_id(claim) or _record_id(claim, index)
        evidence = evidence_by_id.get(evidence_id)
        evidence_text = _record_text(evidence) if evidence is not None else ""
        lexical = _overlap(claim_text, evidence_text)
        numeric = _numeric_agreement(claim_text, evidence_text)
        polarity_mismatch = bool(_NEG_RE.search(claim_text)) != bool(_NEG_RE.search(evidence_text)) if evidence_text else False
        status = _status(lexical, numeric, polarity_mismatch, bool(evidence_text))
        rows.append(
            {
                "claim_id": _record_id(claim, index),
                "evidence_id": evidence_id,
                "alignment_status": status,
                "lexical_overlap": lexical,
                "numeric_agreement": numeric,
                "polarity_mismatch": polarity_mismatch,
            }
        )
    return {"alignments": rows}


def _record_id(record: Any, index: int) -> str:
    return result_id(record, index)


def _record_text(record: Any) -> str:
    if record is None:
        return ""
    return string(value(record, "claim_text")) or string(value(record, "quote")) or content_text(record)


def _evidence_id(claim: Any) -> str | None:
    for key in ("evidence_id", "citation_id", "quote_id", "source_id"):
        text = string(value(claim, key))
        if text:
            return text
    return None


def _overlap(left: str, right: str) -> float:
    left_terms = tokens(left, min_length=4)
    right_terms = tokens(right, min_length=4)
    if not left_terms or not right_terms:
        return 0.0
    return round(len(left_terms & right_terms) / len(left_terms | right_terms), 2)


def _numeric_agreement(left: str, right: str) -> float:
    left_numbers = set(_NUMBER_RE.findall(left or ""))
    right_numbers = set(_NUMBER_RE.findall(right or ""))
    if not left_numbers and not right_numbers:
        return 1.0
    if not left_numbers or not right_numbers:
        return 0.0
    return round(len(left_numbers & right_numbers) / len(left_numbers | right_numbers), 2)


def _status(lexical: float, numeric: float, polarity_mismatch: bool, has_evidence: bool) -> str:
    if not has_evidence or polarity_mismatch or numeric == 0.0:
        return "conflicting"
    if lexical >= 0.35 and numeric >= 0.75:
        return "aligned"
    return "weak"
