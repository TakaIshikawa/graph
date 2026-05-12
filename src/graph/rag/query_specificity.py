"""Analyze how specific a RAG/search query is."""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_QUOTE_RE = re.compile(r'"[^"]+"|\'[^\']+\'')
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{4}|Q[1-4]\s+\d{4})\b", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_ID_RE = re.compile(r"\b(?:[A-Z]{2,}-\d+|[A-Za-z]+_\d+|[0-9a-f]{8,})\b")
_OPERATOR_RE = re.compile(r"\b(?:AND|OR|NOT|site|filetype|after|before):?|\-\"?[A-Za-z0-9]", re.IGNORECASE)
_CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")


def _tokens(query: str) -> list[str]:
    return _TOKEN_RE.findall(query)


def _bounded_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _classification(score: float) -> str:
    if score >= 0.68:
        return "specific"
    if score >= 0.34:
        return "focused"
    return "broad"


def _suggestions(signals: dict[str, Any]) -> list[str]:
    suggestions = []
    if signals["token_count"] < 3:
        suggestions.append("add more topic terms")
    if signals["quoted_phrase_count"] == 0:
        suggestions.append("add an exact phrase")
    if signals["date_count"] == 0:
        suggestions.append("add a date or time range")
    if signals["operator_count"] == 0:
        suggestions.append("add a search operator such as site: or filetype:")
    if signals["entity_like_count"] == 0:
        suggestions.append("name a person, organization, product, or place")
    return suggestions[:4]


def analyze_query_specificity(query: str | None) -> dict[str, Any]:
    """Classify a query as broad, focused, or specific with explainable signals."""
    text = "" if query is None else str(query).strip()
    tokens = _tokens(text)
    token_count = len(tokens)
    quoted_phrases = _QUOTE_RE.findall(text)
    dates = _DATE_RE.findall(text)
    urls = _URL_RE.findall(text)
    concrete_ids = _ID_RE.findall(text)
    operators = _OPERATOR_RE.findall(text)
    entities = [
        match.group(0)
        for match in _CAPITALIZED_RE.finditer(text)
        if match.group(0).casefold() not in {"and", "or", "not"}
    ]
    entities.extend(_ACRONYM_RE.findall(text))

    signals = {
        "token_count": token_count,
        "quoted_phrase_count": len(quoted_phrases),
        "operator_count": len(operators),
        "entity_like_count": len(entities),
        "date_count": len(dates),
        "concrete_id_count": len(concrete_ids) + len(urls),
        "has_url": bool(urls),
    }

    score = 0.0
    score += min(token_count / 8, 1.0) * 0.28
    score += min(len(quoted_phrases), 2) * 0.12
    score += min(len(operators), 2) * 0.08
    score += min(len(entities), 3) * 0.08
    score += min(len(dates), 2) * 0.11
    score += min(signals["concrete_id_count"], 2) * 0.18
    if signals["has_url"]:
        score += 0.12
    score = round(_bounded_score(score), 3)

    specificity = _classification(score)
    return {
        "specificity": specificity,
        "score": score,
        "signals": signals,
        "suggested_refinements": _suggestions(signals) if specificity == "broad" else [],
    }
