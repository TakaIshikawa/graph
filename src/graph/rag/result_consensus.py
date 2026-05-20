"""Group retrieved RAG results by claim text and summarize consensus."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, source_id, string, tokens, value

_CLAIM_KEYS = ("claim", "topic", "title", "snippet", "summary", "content", "text")
_SOURCE_KEYS = ("source_id", "source", "source_name", "source_project", "domain", "url")
_NEGATIVE_RE = re.compile(r"\b(?:not|no|never|false|incorrect|refute[sd]?|den(?:y|ies|ied))\b", re.IGNORECASE)
_POSITIVE_RE = re.compile(r"\b(?:true|correct|confirm(?:s|ed)?|support(?:s|ed)?|available|active)\b", re.IGNORECASE)


def analyze_result_consensus(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return claim groups with source counts and consensus levels."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    grouped: dict[str, list[tuple[Any, int]]] = defaultdict(list)
    for index, result in enumerate(rows):
        claim = _normalized_claim(result)
        if claim is None:
            continue
        grouped[claim].append((result, index))

    groups = []
    for claim in sorted(grouped):
        items = grouped[claim]
        sources = sorted({_source(result) for result, _ in items})
        stances = {_stance(result) for result, _ in items}
        consensus_level = _consensus_level(sources, stances)
        groups.append(
            {
                "normalized_claim": claim,
                "result_ids": [result_id(result, index) for result, index in items],
                "source_count": len(sources),
                "evidence_count": len(items),
                "consensus_level": consensus_level,
                "sources": sources,
            }
        )
    return groups


def _normalized_claim(result: Any) -> str | None:
    for key in _CLAIM_KEYS:
        text = string(value(result, key))
        if text is None:
            continue
        sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
        terms = sorted(tokens(sentence, min_length=2))
        if terms:
            return " ".join(terms)
    return None


def _source(result: Any) -> str:
    for key in _SOURCE_KEYS:
        text = string(value(result, key))
        if text is not None:
            return text
    return source_id(result) or "unknown"


def _stance(result: Any) -> str:
    explicit = string(value(result, "stance")) or string(value(result, "status"))
    if explicit is not None:
        lowered = explicit.casefold()
        if lowered in {"support", "supports", "confirmed", "true", "active", "available"}:
            return "positive"
        if lowered in {"oppose", "opposes", "refuted", "false", "inactive", "unavailable"}:
            return "negative"
        return lowered

    text = " ".join(filter(None, (string(value(result, key)) for key in ("claim", "snippet", "content", "text"))))
    has_positive = bool(_POSITIVE_RE.search(text))
    has_negative = bool(_NEGATIVE_RE.search(text))
    if has_positive and not has_negative:
        return "positive"
    if has_negative and not has_positive:
        return "negative"
    return "neutral"


def _consensus_level(sources: list[str], stances: set[str]) -> str:
    non_neutral = stances.difference({"neutral"})
    if len(non_neutral) > 1 or ("positive" in stances and "negative" in stances):
        return "conflicting"
    if len(sources) > 1:
        return "multi-source"
    return "single-source"
