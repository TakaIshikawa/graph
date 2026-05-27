"""Analyze source reuse risk in retrieved RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import domain_for, rounded_ratio, string, value


def analyze_result_source_reuse_risk(results: Iterable[Any]) -> dict[str, Any]:
    sources = [_source(result) for result in results or []]
    total = len(sources)
    counts = Counter(sources)
    top_source = None
    top_count = 0
    if counts:
        top_source, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    reuse_ratio = rounded_ratio(top_count, total)
    repeated = [{"source": source, "count": count} for source, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count > 1]
    risk = "high" if reuse_ratio >= 0.75 and total else "medium" if reuse_ratio >= 0.5 and total else "low"
    return {
        "total_results": total,
        "unique_sources": len(counts),
        "top_source": top_source,
        "top_source_count": top_count,
        "reuse_ratio": round(reuse_ratio, 3),
        "repeated_sources": repeated,
        "risk_level": risk,
    }


def _source(result: Any) -> str:
    for key in ("source", "source_id", "domain"):
        text = string(value(result, key))
        if text:
            return text.casefold()
    return domain_for(result) or "unknown"
