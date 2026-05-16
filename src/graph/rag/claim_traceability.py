"""Map drafted answer claims to supporting RAG result snippets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import (
    MISSING,
    content_text,
    iter_strings,
    result_id,
    source_id,
    string,
    tokens,
    value,
)


def _claim_text(claim: Any) -> str:
    if isinstance(claim, str):
        return " ".join(claim.split())
    for key in ("claim", "text", "content", "sentence"):
        text = string(value(claim, key))
        if text is not None:
            return text
    return string(claim) or ""


def _claim_id(claim: Any, index: int) -> str:
    for key in ("id", "claim_id"):
        text = string(value(claim, key))
        if text is not None:
            return text
    return f"claim-{index + 1}"


def _claim_refs(claim: Any) -> set[str]:
    refs: set[str] = set()
    for key in ("citation_ids", "citations", "source_ids", "sources", "result_ids"):
        raw = value(claim, key)
        if raw is MISSING:
            continue
        refs.update(text for text in iter_strings(raw) if text)
    return refs


def _result_refs(result: Any, rid: str) -> set[str]:
    refs = {rid}
    sid = source_id(result)
    if sid:
        refs.add(sid)
    for key in ("citation_id", "citation_ids", "citations", "source_id", "source_ids"):
        refs.update(text for text in iter_strings(value(result, key)) if text)
    return refs


def map_claim_traceability(claims: Iterable[Any], results: Iterable[Any]) -> dict[str, Any]:
    """Return support matches connecting claims to result evidence."""
    try:
        claim_rows = list(claims or [])
    except TypeError:
        claim_rows = []
    try:
        result_rows = list(results or [])
    except TypeError:
        result_rows = []

    normalized_results = []
    for index, result in enumerate(result_rows):
        rid = result_id(result, index)
        normalized_results.append(
            {
                "result": result,
                "result_id": rid,
                "source_id": source_id(result),
                "refs": _result_refs(result, rid),
                "tokens": tokens(content_text(result)),
            }
        )

    claim_support: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, str]] = []
    explicit_count = 0
    overlap_count = 0

    for index, claim in enumerate(claim_rows):
        cid = _claim_id(claim, index)
        text = _claim_text(claim)
        claim_terms = tokens(text)
        refs = _claim_refs(claim)
        matches: list[dict[str, Any]] = []

        for result in normalized_results:
            explicit_refs = refs & result["refs"]
            if explicit_refs:
                matches.append(
                    {
                        "match_type": "citation",
                        "score": 1.0,
                        "result_id": result["result_id"],
                        "source_id": result["source_id"],
                    }
                )
                explicit_count += 1
                continue

            if not claim_terms or not result["tokens"]:
                continue
            overlap = claim_terms & result["tokens"]
            score = len(overlap) / len(claim_terms)
            if score >= 0.4:
                matches.append(
                    {
                        "match_type": "content_overlap",
                        "score": round(score, 4),
                        "result_id": result["result_id"],
                        "source_id": result["source_id"],
                    }
                )
                overlap_count += 1

        matches.sort(key=lambda item: (-float(item["score"]), item["result_id"]))
        record = {
            "claim_id": cid,
            "claim": text,
            "matches": matches,
            "supported": bool(matches),
        }
        claim_support.append(record)
        if not matches:
            unsupported_claims.append({"claim_id": cid, "claim": text})

    return {
        "claims": claim_support,
        "unsupported_claims": unsupported_claims,
        "support_counts": {
            "claim_count": len(claim_rows),
            "result_count": len(result_rows),
            "supported_count": len(claim_rows) - len(unsupported_claims),
            "unsupported_count": len(unsupported_claims),
            "explicit_match_count": explicit_count,
            "overlap_match_count": overlap_count,
        },
    }
