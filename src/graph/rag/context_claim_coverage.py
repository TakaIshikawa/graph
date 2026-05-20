"""Compare claim terms against retrieved RAG context."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, content_text, iter_strings, result_id, rounded_ratio, string, tokens, value


def analyze_context_claim_coverage(claims: Iterable[Any], context_results: Iterable[Any]) -> dict[str, Any]:
    """Return per-claim coverage rows with supporting result IDs."""
    contexts = list(context_results)
    context_terms = [tokens(" ".join([content_text(result), " ".join(iter_strings(value(result, "metadata")))]), min_length=3) for result in contexts]
    rows = []
    for index, claim in enumerate(claims):
        claim_id, text = _claim(claim, index)
        terms = tokens(text, min_length=3)
        covered = set().union(*(terms & result_terms for result_terms in context_terms)) if terms else set()
        supporting = [
            result_id(result, result_index)
            for result_index, (result, result_terms) in enumerate(zip(contexts, context_terms, strict=True))
            if terms and terms & result_terms
        ]
        ratio = rounded_ratio(len(covered), len(terms))
        warnings = []
        if not supporting:
            warnings.append("missing_support")
        elif ratio < 0.75:
            warnings.append("weak_support")
        rows.append(
            {
                "claim_id": claim_id,
                "claim": text,
                "supporting_result_ids": supporting,
                "coverage_ratio": ratio,
                "uncovered_terms": sorted(terms - covered),
                "warnings": warnings,
            }
        )
    return {"claim_count": len(rows), "claims": rows, "warnings": sorted({warning for row in rows for warning in row["warnings"]})}


def _claim(claim: Any, index: int) -> tuple[str, str]:
    claim_id = string(value(claim, "id")) or string(value(claim, "claim_id")) or f"claim-{index + 1}"
    text = string(value(claim, "text"))
    if text is None:
        text = string(value(claim, "claim")) if value(claim, "claim") is not MISSING else string(claim)
    return claim_id, text or ""
