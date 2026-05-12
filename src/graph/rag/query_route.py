"""Suggest deterministic retrieval routes for natural-language RAG queries."""

from __future__ import annotations

import re
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_QUOTE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_TAG_RE = re.compile(r"(?<!\w)(?:#|tag:)([a-zA-Z0-9][\w./-]*)")
_SOURCE_RE = re.compile(
    r"\b(?:source|project|from|in)\s*:\s*([a-zA-Z0-9][\w./-]*)|\b(?:source|project)\s+([a-zA-Z0-9][\w./-]*)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{4}\b")
_DATE_WORDS = {
    "today",
    "yesterday",
    "recent",
    "recently",
    "latest",
    "newest",
    "older",
    "before",
    "after",
    "since",
    "between",
    "week",
    "month",
    "year",
}
_CONTRADICTION_TERMS = {
    "contradict",
    "contradiction",
    "contradictions",
    "conflict",
    "conflicts",
    "disagree",
    "disagrees",
    "disputed",
    "disputes",
    "tension",
    "inconsistent",
    "versus",
    "vs",
}


def _normalize_query(query: Any) -> str:
    if query is None:
        return ""
    return " ".join(str(query).strip().split())


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in COMMON_STOPWORDS
    ]


def _raw_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.casefold()) if len(token) > 1]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip().casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _quoted_phrases(query: str) -> list[str]:
    phrases = []
    for match in _QUOTE_RE.finditer(query):
        phrase = match.group(1) or match.group(2)
        normalized = " ".join(phrase.strip().split())
        if normalized:
            phrases.append(normalized.casefold())
    return _dedupe(phrases)


def _source_hints(query: str) -> list[str]:
    hints: list[str] = []
    for match in _SOURCE_RE.finditer(query):
        value = match.group(1) or match.group(2)
        if value:
            hints.append(value)
    return _dedupe(hints)


def _date_hints(query: str, tokens: list[str]) -> list[str]:
    hints = _ISO_DATE_RE.findall(query)
    hints.extend(token for token in tokens if token in _DATE_WORDS)
    return _dedupe(hints)


def _route(name: str, confidence: float, rationale: str) -> dict[str, Any]:
    return {
        "route": name,
        "confidence": round(confidence, 3),
        "rationale": rationale,
    }


def suggest_query_routes(query: Any) -> dict[str, Any]:
    """Classify a query into ordered retrieval route suggestions."""
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return {
            "normalized_query": "",
            "routes": [],
            "extracted_tags": [],
            "source_hints": [],
            "date_hints": [],
            "rationale": "Blank or invalid query; no retrieval routes suggested.",
        }

    tokens = _tokens(normalized_query)
    extracted_tags = _dedupe(_TAG_RE.findall(normalized_query))
    source_hints = _source_hints(normalized_query)
    date_hints = _date_hints(normalized_query, _raw_tokens(normalized_query))
    quoted_phrases = _quoted_phrases(normalized_query)
    contradiction_terms = sorted(set(tokens) & _CONTRADICTION_TERMS)

    routes = [
        _route(
            "semantic",
            0.5 + min(len(tokens), 8) * 0.02,
            "Use semantic retrieval for the remaining natural-language terms.",
        )
    ]
    if quoted_phrases:
        routes.append(
            _route(
                "exact_title",
                0.94,
                f"Quoted phrase suggests exact title matching: {', '.join(quoted_phrases)}.",
            )
        )
    if extracted_tags:
        routes.append(
            _route(
                "tag_filter",
                0.9,
                f"Tag markers constrain retrieval to: {', '.join(extracted_tags)}.",
            )
        )
    if source_hints:
        routes.append(
            _route(
                "source_filter",
                0.86,
                f"Source or project hints constrain retrieval to: {', '.join(source_hints)}.",
            )
        )
    if date_hints:
        routes.append(
            _route(
                "date_filter",
                0.82,
                f"Date language constrains retrieval by time: {', '.join(date_hints)}.",
            )
        )
    if contradiction_terms:
        routes.append(
            _route(
                "contradiction_check",
                0.88,
                f"Contradiction language asks for disagreement analysis: {', '.join(contradiction_terms)}.",
            )
        )

    routes.sort(key=lambda item: (-float(item["confidence"]), str(item["route"])))
    rationale_parts = [route["rationale"] for route in routes]

    return {
        "normalized_query": normalized_query,
        "routes": routes,
        "extracted_tags": extracted_tags,
        "source_hints": source_hints,
        "date_hints": date_hints,
        "rationale": " ".join(rationale_parts),
    }
