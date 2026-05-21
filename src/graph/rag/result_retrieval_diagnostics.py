"""Diagnose common retrieval quality issues in RAG result lists."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, domain_for, result_id, source_id, tokens


def diagnose_result_retrieval(query: str, results: Iterable[Any]) -> dict[str, Any]:
    """Return aggregate and per-result retrieval diagnostics."""
    query_terms = tokens(query, min_length=3)
    rows = []
    seen_signatures: Counter[frozenset[str]] = Counter()
    for index, result in enumerate(list(results or [])):
        rid = result_id(result, index)
        text = content_text(result)
        result_terms = tokens(text, min_length=3)
        covered = sorted(query_terms & result_terms)
        signature = frozenset(list(result_terms)[:20])
        seen_signatures[signature] += 1
        warnings = []
        if not text:
            warnings.append("missing_content")
        if query_terms and len(covered) / len(query_terms) < 0.34:
            warnings.append("weak_query_coverage")
        if not (source_id(result) or domain_for(result)):
            warnings.append("missing_source_metadata")
        if len(result_terms) < 8:
            warnings.append("shallow_evidence")
        rows.append(
            {
                "result_id": rid,
                "query_terms_matched": covered,
                "query_coverage_ratio": 0.0 if not query_terms else round(len(covered) / len(query_terms), 4),
                "content_token_count": len(result_terms),
                "source": source_id(result),
                "domain": domain_for(result),
                "warnings": warnings,
                "_signature": signature,
            }
        )

    duplicate_ids = set()
    signatures_by_id = {row["result_id"]: row["_signature"] for row in rows}
    for row in rows:
        signature = row.pop("_signature")
        if signature and seen_signatures[signature] > 1:
            row["warnings"].append("duplicate_like_result")
            duplicate_ids.add(row["result_id"])

    aggregate_warnings = sorted({warning for row in rows for warning in row["warnings"]})
    return {
        "result_count": len(rows),
        "query_term_count": len(query_terms),
        "missing_content_count": sum("missing_content" in row["warnings"] for row in rows),
        "missing_source_metadata_count": sum("missing_source_metadata" in row["warnings"] for row in rows),
        "duplicate_like_count": len(duplicate_ids),
        "per_result": rows[:50],
        "warnings": aggregate_warnings,
    }
