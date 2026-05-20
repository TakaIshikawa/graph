"""Plan source requirements for RAG answers from query and result signals."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import domain_for, source_id, string, value

_COMPARISON_RE = re.compile(r"\b(?:compare|versus|vs\.?|difference|trade[-\s]?off|which\s+is\s+better)\b", re.IGNORECASE)
_CURRENT_RE = re.compile(r"\b(?:latest|current|currently|recent|today|newest|up\s+to\s+date)\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\b(?:cite|citation|source|sources|reference|references|with\s+evidence)\b", re.IGNORECASE)
_QUANT_RE = re.compile(r"\b(?:\d+(?:\.\d+)?%?|percent|statistic|metric|rate|count|average|median|trend)\b", re.IGNORECASE)
_HIGH_STAKES_RE = re.compile(
    r"\b(?:medical|health|clinical|diagnosis|legal|law|lawsuit|contract|financial|finance|tax|investment|loan|insurance)\b",
    re.IGNORECASE,
)
_SOURCE_KEYS = ("source_id", "source", "source_name", "source_project", "domain", "url", "source_url")


def plan_answer_source_requirements(query: str, results: Iterable[Any] | None = None) -> dict[str, Any]:
    """Return deterministic source-count, citation, and diversity requirements."""
    normalized_query = " ".join(str(query).split())
    reasons = _query_reasons(normalized_query)

    minimum_sources = 1
    if "citation_requested" in reasons:
        minimum_sources = max(minimum_sources, 2)
    if "latest_or_current_query" in reasons:
        minimum_sources = max(minimum_sources, 2)
    if "quantitative_query" in reasons:
        minimum_sources = max(minimum_sources, 2)
    if "comparison_query" in reasons:
        minimum_sources = max(minimum_sources, 3)
    if "medical_legal_or_financial_query" in reasons:
        minimum_sources = max(minimum_sources, 3)

    require_citations = bool(
        {"citation_requested", "latest_or_current_query", "quantitative_query", "medical_legal_or_financial_query"}.intersection(reasons)
    )
    require_source_diversity = bool(
        {"comparison_query", "latest_or_current_query", "medical_legal_or_financial_query"}.intersection(reasons)
    )

    available_sources = _available_sources(results)
    source_gap = available_sources is not None and len(available_sources) < minimum_sources

    return {
        "normalized_query": normalized_query,
        "minimum_sources": minimum_sources,
        "require_citations": require_citations,
        "require_source_diversity": require_source_diversity,
        "source_gap": source_gap,
        "available_source_count": None if available_sources is None else len(available_sources),
        "reasons": reasons,
    }


def _query_reasons(query: str) -> list[str]:
    checks = (
        ("comparison_query", _COMPARISON_RE),
        ("latest_or_current_query", _CURRENT_RE),
        ("medical_legal_or_financial_query", _HIGH_STAKES_RE),
        ("citation_requested", _CITATION_RE),
        ("quantitative_query", _QUANT_RE),
    )
    return [label for label, pattern in checks if pattern.search(query)]


def _available_sources(results: Iterable[Any] | None) -> set[str] | None:
    if results is None:
        return None
    try:
        rows = list(results)
    except TypeError:
        return set()

    sources: set[str] = set()
    for result in rows:
        source = _source(result)
        if source is not None:
            sources.add(source)
    return sources


def _source(result: Any) -> str | None:
    for key in _SOURCE_KEYS:
        text = string(value(result, key))
        if text is not None:
            return text
    return source_id(result) or domain_for(result)
