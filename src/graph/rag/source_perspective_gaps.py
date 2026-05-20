"""Classify RAG source perspectives and report query-driven gaps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, domain_for, result_id, string, value

_PERSPECTIVES = ("vendor", "user_community", "official", "academic", "news_media", "internal_note", "unknown")


def analyze_source_perspective_gaps(query: Any, results: Iterable[Any]) -> dict[str, Any]:
    """Return source perspective counts and missing recommended perspectives."""
    rows = []
    counts: Counter[str] = Counter()
    for index, result in enumerate(results):
        perspective = _perspective(result)
        counts[perspective] += 1
        rows.append({"result_id": result_id(result, index), "perspective": perspective})

    required = _required_perspectives(string(query) or "")
    missing = [item for item in required if counts[item] == 0]
    warnings = []
    if counts["unknown"]:
        warnings.append("unknown_perspectives")
    if missing:
        warnings.append("missing_recommended_perspectives")
    return {
        "perspectives": rows,
        "counts": {key: counts[key] for key in _PERSPECTIVES if counts[key]},
        "missing_perspectives": missing,
        "warnings": warnings,
    }


def _perspective(result: Any) -> str:
    explicit = string(value(result, "perspective")) or string(value(result, "source_type"))
    if explicit:
        normalized = explicit.casefold().replace("-", "_").replace(" ", "_")
        aliases = {"community": "user_community", "user": "user_community", "news": "news_media", "media": "news_media"}
        normalized = aliases.get(normalized, normalized)
        if normalized in _PERSPECTIVES:
            return normalized
    text = f"{content_text(result)} {domain_for(result) or ''}".casefold()
    if any(cue in text for cue in ("github", "reddit", "forum", "community", "stackoverflow")):
        return "user_community"
    if any(cue in text for cue in (".gov", "official", "docs.", "documentation")):
        return "official"
    if any(cue in text for cue in ("journal", "university", "arxiv", "doi", "study")):
        return "academic"
    if any(cue in text for cue in ("news", "press", "reuters", "apnews")):
        return "news_media"
    if any(cue in text for cue in ("internal", "memo", "private note")):
        return "internal_note"
    if any(cue in text for cue in ("vendor", "pricing", "product", "sales")):
        return "vendor"
    return "unknown"


def _required_perspectives(query: str) -> list[str]:
    lowered = query.casefold()
    if any(cue in lowered for cue in ("recommend", "best", "should", "compare", "versus", " vs ")):
        return ["user_community", "academic", "news_media"]
    if any(cue in lowered for cue in ("policy", "legal", "official")):
        return ["official"]
    return []
