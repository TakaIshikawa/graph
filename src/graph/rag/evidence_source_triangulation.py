"""Score whether evidence claims are triangulated across distinct sources."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, domain_for, result_id, string, value

_SENTENCE_RE = re.compile(r"[^.!?\n]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SOURCE_KEYS = ("source", "source_id", "url", "domain", "publisher", "author")


def score_evidence_source_triangulation(results: Iterable[Any], *, min_sources: int = 2) -> dict[str, Any]:
    """Group claim-like snippets and flag those backed by too few sources."""
    if not isinstance(min_sources, int) or isinstance(min_sources, bool) or min_sources < 1:
        raise ValueError("min_sources must be a positive integer")
    rows = list(results or [])
    grouped: dict[str, set[str]] = defaultdict(set)
    result_rows = []
    source_counter: Counter[str] = Counter()

    for index, result in enumerate(rows):
        source = _source_identity(result)
        source_counter[source] += 1
        claim_key = _claim_key(content_text(result))
        grouped[claim_key].add(source)
        result_rows.append(
            {
                "result_id": result_id(result, index),
                "claim_key": claim_key,
                "source": source,
                "reasons": [] if source != "unknown_source" else ["missing_source_identity"],
            }
        )

    for row in result_rows:
        if len(grouped[row["claim_key"]]) < min_sources:
            row["reasons"].append("insufficient_distinct_sources")

    reason_counts = _reason_counts(result_rows)
    return {
        "total_results": len(rows),
        "distinct_sources": len(source_counter),
        "triangulated_claims": sum(1 for sources in grouped.values() if len(sources) >= min_sources),
        "single_source_claims": sum(1 for sources in grouped.values() if len(sources) < min_sources),
        "source_counts": dict(sorted(source_counter.items())),
        "results": result_rows,
        "reason_counts": reason_counts,
        "warnings": ["no_results"] if not rows else (["single_source_claims"] if any(row["reasons"] for row in result_rows) else []),
    }


def _source_identity(result: Any) -> str:
    domain = domain_for(result)
    if domain:
        return domain
    for key in _SOURCE_KEYS:
        text = string(value(result, key))
        if text:
            return text.casefold()
    return "unknown_source"


def _claim_key(text: str) -> str:
    sentence = (_SENTENCE_RE.findall(text) or [text or "missing_text"])[0]
    tokens = _TOKEN_RE.findall(sentence.casefold())
    return " ".join(tokens[:14]) or "missing_text"


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter(reason for row in rows for reason in row["reasons"])
    return dict(sorted(counter.items()))
