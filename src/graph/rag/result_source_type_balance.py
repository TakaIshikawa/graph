"""Analyze source-type balance in RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import domain_for, string, value

_TYPES = ("documentation", "paper", "news", "blog", "forum", "code", "dataset", "video", "unknown")


def analyze_result_source_type_balance(results: Iterable[Any]) -> dict[str, Any]:
    items = list(results or [])
    counts = Counter(_source_type(item) for item in items)
    dominant = max(_TYPES, key=lambda label: (counts[label], -_TYPES.index(label))) if items else "unknown"
    diversity = round((len([v for v in counts.values() if v]) - 1) / (len(_TYPES) - 1), 4) if items else 0.0
    return {
        "total_results": len(items),
        "source_type_counts": dict(sorted(counts.items())),
        "dominant_source_type": dominant,
        "diversity_score": diversity,
    }


def _source_type(result: Any) -> str:
    for key in ("source_type", "type"):
        text = string(value(result, key))
        if text and text.casefold().replace("_", " ") in _TYPES:
            return text.casefold().replace("_", " ")
    domain = domain_for(result) or ""
    url = string(value(result, "url")) or ""
    haystack = f"{domain} {url}".casefold()
    if "docs." in haystack or "/docs" in haystack or "developer." in haystack:
        return "documentation"
    if "arxiv.org" in haystack or "doi.org" in haystack or "pubmed" in haystack:
        return "paper"
    if "news" in haystack or "nytimes" in haystack or "reuters" in haystack:
        return "news"
    if "blog" in haystack or "medium.com" in haystack:
        return "blog"
    if "forum" in haystack or "reddit.com" in haystack or "stackoverflow.com" in haystack:
        return "forum"
    if "github.com" in haystack or "gitlab.com" in haystack:
        return "code"
    if "kaggle.com" in haystack or "data.gov" in haystack or "dataset" in haystack:
        return "dataset"
    if "youtube.com" in haystack or "vimeo.com" in haystack:
        return "video"
    return "unknown"
