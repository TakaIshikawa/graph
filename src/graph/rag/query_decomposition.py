"""Decompose compound user queries into retrieval-ready subqueries."""

from __future__ import annotations

import re
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE
from graph.rag.query_intent import classify_query_intent

_QUOTE_RE = re.compile(r'"([^"]+)"')
_YEAR_RE = re.compile(r"(?<![A-Za-z0-9])((?:19|20)\d{2})(?![A-Za-z0-9])")
_DATE_OPERATOR_RE = re.compile(
    r"\b(before|after|since|until|through|during|from)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(
    r"\b(?:source|from)\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)",
    re.IGNORECASE,
)
_SOURCE_STOPWORDS = frozenset({"before", "after", "since", "until", "through", "during", "from"})
_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|contrast|versus|vs\.?|difference between|differences between|tradeoffs?|better than|worse than)\b",
    re.IGNORECASE,
)
_BOUNDARY_RE = re.compile(
    r"\s+(?:and|also|plus|as well as|while|but|versus|vs\.?|compared with|compared to)\s+",
    re.IGNORECASE,
)
_PROTECTED_RE = re.compile(r"__QUOTE_(\d+)__")


def decompose_query_for_retrieval(query: object, *, max_subqueries: int = 5) -> dict[str, Any]:
    """Split one user query into focused retrieval subqueries and constraints."""
    max_value = _validate_max_subqueries(max_subqueries)
    original_query = _normalize_text(query)
    constraints = _detected_constraints(original_query)
    clauses = _split_clauses(original_query)
    if not clauses:
        clauses = [original_query] if original_query else []
    clauses = clauses[:max_value]

    subqueries = [_subquery(clause, index, len(clauses), constraints) for index, clause in enumerate(clauses)]
    strategy = {
        "mode": "single_clause" if len(subqueries) <= 1 else "decomposed",
        "subquery_count": len(subqueries),
        "split_on": _split_strategy(original_query, len(subqueries)),
        "max_subqueries": max_value,
    }
    return {
        "original_query": original_query,
        "subqueries": subqueries,
        "detected_constraints": constraints,
        "strategy": strategy,
    }


def _validate_max_subqueries(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_subqueries must be a positive integer")
    return value


def _subquery(clause: str, index: int, total: int, constraints: dict[str, Any]) -> dict[str, Any]:
    intent_payload = classify_query_intent(clause)
    required_terms = _required_terms(clause)
    optional_terms = _optional_terms(clause, required_terms, constraints)
    return {
        "text": clause,
        "intent": intent_payload["intent"],
        "required_terms": required_terms,
        "optional_terms": optional_terms,
        "rationale": _rationale(clause, index, total),
    }


def _split_clauses(query: str) -> list[str]:
    if not query:
        return []
    protected, quoted = _protect_quotes(query)
    protected = _COMPARISON_RE.sub(lambda match: f" {match.group(0)} ", protected)
    parts = [part.strip(" ,;:") for part in _BOUNDARY_RE.split(protected)]
    clauses = [_restore_quotes(part, quoted) for part in parts if _restore_quotes(part, quoted)]
    return _dedupe_preserving_order(clauses)


def _protect_quotes(query: str) -> tuple[str, list[str]]:
    quoted: list[str] = []

    def replace(match: re.Match[str]) -> str:
        quoted.append(match.group(1))
        return f"__QUOTE_{len(quoted) - 1}__"

    return _QUOTE_RE.sub(replace, query), quoted


def _restore_quotes(value: str, quoted: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return f'"{quoted[index]}"'

    return _normalize_text(_PROTECTED_RE.sub(replace, value))


def _detected_constraints(query: str) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    quoted_phrases = sorted({_normalize_text(match.group(1)) for match in _QUOTE_RE.finditer(query)})
    if quoted_phrases:
        constraints["quoted_phrases"] = quoted_phrases

    years = sorted({match.group(1) for match in _YEAR_RE.finditer(query)})
    date_constraints: dict[str, Any] = {}
    if years:
        date_constraints["years"] = years
    for operator, year in _DATE_OPERATOR_RE.findall(query):
        normalized_operator = operator.casefold()
        if normalized_operator in {"until", "through"}:
            normalized_operator = "before"
        if normalized_operator == "from":
            normalized_operator = "after"
        date_constraints[normalized_operator] = year
    if date_constraints:
        constraints["date"] = date_constraints

    sources = sorted(
        {
            source.casefold()
            for source in _SOURCE_RE.findall(query)
            if source.casefold() not in _SOURCE_STOPWORDS and not _YEAR_RE.fullmatch(source)
        }
    )
    if sources:
        constraints["sources"] = sources

    entities = _entity_terms(query)
    if entities:
        constraints["entity_terms"] = entities

    return constraints


def _required_terms(clause: str) -> list[str]:
    quoted = [_normalize_text(match.group(1)) for match in _QUOTE_RE.finditer(clause)]
    searchable_clause = _QUOTE_RE.sub(" ", clause).casefold()
    tokens = [
        token
        for token in TOKEN_RE.findall(searchable_clause)
        if len(token) > 1 and token not in COMMON_STOPWORDS
    ]
    terms = quoted + tokens
    return _dedupe_preserving_order(terms)


def _optional_terms(clause: str, required_terms: list[str], constraints: dict[str, Any]) -> list[str]:
    required = {term.casefold() for term in required_terms}
    optional = [
        term
        for term in constraints.get("entity_terms", [])
        if term.casefold() not in required and term.casefold() in clause.casefold()
    ]
    return _dedupe_preserving_order(optional)


def _entity_terms(query: str) -> list[str]:
    terms: list[str] = []
    for phrase in _QUOTE_RE.findall(query):
        terms.append(_normalize_text(phrase))
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b", query):
        text = _normalize_text(match.group(0))
        if text.casefold() not in COMMON_STOPWORDS and not _YEAR_RE.fullmatch(text):
            terms.append(text)
    return sorted(set(terms), key=lambda term: (term.casefold(), term))


def _rationale(clause: str, index: int, total: int) -> str:
    if total <= 1:
        return "single focused retrieval query"
    if _COMPARISON_RE.search(clause):
        return "comparison cue retained in focused clause"
    return f"clause {index + 1} of compound query"


def _split_strategy(query: str, count: int) -> list[str]:
    if count <= 1:
        return []
    strategies: list[str] = []
    if _BOUNDARY_RE.search(query):
        strategies.append("conjunction")
    if _COMPARISON_RE.search(query):
        strategies.append("comparison")
    return strategies or ["clause_boundary"]


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())
