"""Audit whether answers state the breadth of retrieved evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_PHRASES = ("across sources", "the retrieved evidence", "available records", "most sources", "some sources", "no retrieved source")


def audit_answer_evidence_coverage_statement(answer: str, evidence: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    records = list(evidence or ())
    text = str(answer or "").lower()
    matched = [phrase for phrase in _PHRASES if re.search(rf"\b{re.escape(phrase)}\b", text)]
    return {
        "evidence_count": len(records),
        "has_coverage_statement": bool(matched),
        "missing_coverage_statement": len(records) > 1 and not matched,
        "matched_phrases": matched,
        "samples": records[:sample_limit],
    }
