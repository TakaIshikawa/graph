"""Analyze claim corroboration gaps across evidence results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, domain_for, result_id, source_id, tokens


def analyze_evidence_corroboration_gaps(claims: Iterable[str], results: Iterable[Any]) -> dict[str, Any]:
    """Estimate independent support for each claim using token overlap."""
    result_rows = []
    for index, result in enumerate(list(results or [])):
        result_rows.append(
            {
                "id": result_id(result, index),
                "tokens": tokens(content_text(result), min_length=3),
                "source": source_id(result) or domain_for(result) or "unknown",
            }
        )
    claim_rows = []
    for claim in claims or []:
        claim_text = " ".join(str(claim or "").split())
        claim_terms = tokens(claim_text, min_length=3)
        supporting = [row for row in result_rows if _supports(claim_terms, row["tokens"])]
        sources = sorted({row["source"] for row in supporting})
        warnings = []
        if not supporting:
            warnings.append("unsupported_claim")
        elif len(sources) == 1:
            warnings.append("single_source_support")
        claim_rows.append(
            {
                "claim": claim_text,
                "support_count": len(supporting),
                "independent_source_count": len(sources),
                "supporting_result_ids": [row["id"] for row in supporting],
                "supporting_sources": sources,
                "warnings": warnings,
            }
        )
    unsupported = sum("unsupported_claim" in row["warnings"] for row in claim_rows)
    single = sum("single_source_support" in row["warnings"] for row in claim_rows)
    warnings = []
    if claim_rows and unsupported / len(claim_rows) >= 0.5:
        warnings.append("unsupported_claims_high")
    if claim_rows and single / len(claim_rows) >= 0.5:
        warnings.append("single_source_claims_high")
    return {"claims": claim_rows, "warnings": warnings}


def _supports(claim_terms: set[str], result_terms: set[str]) -> bool:
    if not claim_terms:
        return False
    overlap = claim_terms & result_terms
    return len(overlap) >= min(3, len(claim_terms)) or len(overlap) / len(claim_terms) >= 0.6
