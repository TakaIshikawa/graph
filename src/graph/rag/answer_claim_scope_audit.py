"""Audit whether answer claims overstate scope relative to retrieved evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_BROAD_SCOPE_RE = re.compile(
    r"\b(always|never|all|none|everyone|everybody|proven|guaranteed|universally|must|only)\b",
    re.I,
)
_NARROW_SCOPE_RE = re.compile(
    r"\b(some|may|might|can|could|sample|pilot|limited|early|preliminary|small|subset|case study|observed)\b",
    re.I,
)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def audit_answer_claim_scope(answer: str, results: Iterable[Any] = ()) -> dict[str, Any]:
    """Return deterministic scope-overstatement rows for broad answer claims."""
    answer_text = str(answer or "").strip()
    rows = list(results or [])
    evidence = _evidence_rows(rows)
    warnings = []
    if not answer_text:
        warnings.append("no_answer")
    if not rows:
        warnings.append("no_results")
    if not answer_text:
        return {
            "total_claims": 0,
            "scoped_claims": 0,
            "overstated_claims": 0,
            "claims": [],
            "reason_counts": {},
            "warnings": warnings,
        }

    claims = []
    for index, sentence in enumerate(_sentences(answer_text)):
        cues = sorted({match.group(1).casefold() for match in _BROAD_SCOPE_RE.finditer(sentence)})
        if not cues:
            continue
        matched = [row for row in evidence if any(cue in row["text"].casefold() for cue in cues)]
        narrowing = [row for row in evidence if _NARROW_SCOPE_RE.search(row["text"])]
        reasons = []
        if not matched:
            reasons.append("missing_matching_scope_support")
        if narrowing:
            reasons.append("evidence_narrows_scope")
        status = "overstated" if reasons else "scoped"
        claims.append(
            {
                "sentence_index": index,
                "claim": sentence[:200],
                "scope_cues": cues,
                "scope_status": status,
                "reasons": reasons,
                "matched_result_ids": [row["result_id"] for row in matched],
                "narrowing_result_ids": [row["result_id"] for row in narrowing],
            }
        )

    reason_counts = _reason_counts(claims)
    if any(claim["scope_status"] == "overstated" for claim in claims):
        warnings.append("overstated_scope_claims")
    return {
        "total_claims": len(claims),
        "scoped_claims": sum(1 for claim in claims if claim["scope_status"] == "scoped"),
        "overstated_claims": sum(1 for claim in claims if claim["scope_status"] == "overstated"),
        "claims": claims,
        "reason_counts": reason_counts,
        "warnings": warnings,
    }


def _evidence_rows(results: list[Any]) -> list[dict[str, str]]:
    return [
        {"result_id": result_id(result, index), "text": content_text(result)}
        for index, result in enumerate(results)
        if content_text(result)
    ]


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _reason_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter(reason for claim in claims for reason in claim["reasons"])
    return dict(sorted(counter.items()))
