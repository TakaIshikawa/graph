"""Detect deterministic ambiguity signals in natural-language RAG queries."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_BROAD_NOUNS = {
    "approach",
    "area",
    "case",
    "data",
    "effect",
    "impact",
    "information",
    "issue",
    "method",
    "overview",
    "policy",
    "problem",
    "process",
    "project",
    "result",
    "strategy",
    "system",
    "thing",
    "topic",
    "trend",
}
_VAGUE_PRONOUNS = {
    "it",
    "its",
    "that",
    "this",
    "these",
    "they",
    "them",
    "those",
}
_RELATIVE_DATES = {
    "current",
    "currently",
    "latest",
    "recent",
    "recently",
    "today",
    "tomorrow",
    "tonight",
    "upcoming",
    "yesterday",
}
_QUESTION_FANOUT = {
    "and",
    "compare",
    "comparison",
    "contrast",
    "differences",
    "or",
    "pros",
    "versus",
    "vs",
}
_QUESTION_WORDS = {"how", "what", "when", "where", "which", "who", "why"}
_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*\b")


def _validate_max_terms(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_terms must be a non-negative integer")
    return value


def _normalize_query(query: Any) -> str:
    if query is None:
        return ""
    return " ".join(str(query).strip().split())


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _known_terms(known_terms: Iterable[Any] | None) -> set[str]:
    if known_terms is None:
        return set()
    return {
        " ".join(str(term).strip().casefold().split())
        for term in known_terms
        if str(term).strip()
    }


def _known_words(known: set[str]) -> set[str]:
    words: set[str] = set()
    for term in known:
        words.update(TOKEN_RE.findall(term))
    return words


def _signal(signal_type: str, terms: Iterable[str], weight: float) -> dict[str, Any]:
    stable_terms = sorted(set(terms), key=lambda item: (item.casefold(), item))
    return {
        "type": signal_type,
        "terms": stable_terms,
        "count": len(stable_terms),
        "weight": weight,
    }


def _unknown_capitalized_terms(query: str, known: set[str]) -> list[str]:
    terms = []
    current: list[str] = []
    def flush() -> None:
        if not current:
            return
        value = " ".join(current)
        words = value.split()
        while words and (
            words[0].casefold() in COMMON_STOPWORDS
            or words[0].casefold() in _QUESTION_WORDS
            or words[0].casefold() in _QUESTION_FANOUT
        ):
            words.pop(0)
        value = " ".join(words)
        if value and value.casefold() not in known:
            first_token = value.split()[0].casefold()
            if first_token not in COMMON_STOPWORDS and first_token not in _QUESTION_WORDS:
                terms.append(value)
        current.clear()

    for raw in re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]+", query):
        if _CAPITALIZED_WORD_RE.fullmatch(raw):
            current.append(raw)
            continue
        flush()
    flush()
    return sorted(set(terms), key=lambda item: (item.casefold(), item))


def detect_query_ambiguity(
    query: Any,
    *,
    known_terms: Iterable[Any] | None = None,
    max_terms: int = 8,
) -> dict[str, Any]:
    """Return deterministic ambiguity signals for a query before retrieval."""
    max_terms_value = _validate_max_terms(max_terms)
    normalized = _normalize_query(query)
    tokens = _tokens(normalized)
    token_set = set(tokens)
    known = _known_terms(known_terms)
    known_words = _known_words(known)

    signals = []
    vague_pronouns = sorted(token_set & _VAGUE_PRONOUNS)
    if vague_pronouns:
        signals.append(_signal("vague_pronoun", vague_pronouns, 0.2))

    relative_dates = sorted(token_set & _RELATIVE_DATES)
    if relative_dates:
        signals.append(_signal("relative_date", relative_dates, 0.22))

    broad_nouns = sorted((token_set & _BROAD_NOUNS) - known_words)
    if broad_nouns:
        signals.append(_signal("broad_noun", broad_nouns, 0.18))

    fanout_terms = sorted(token_set & _QUESTION_FANOUT)
    question_count = sum(1 for token in tokens if token in _QUESTION_WORDS)
    question_marks = normalized.count("?")
    if fanout_terms or question_count > 1 or question_marks > 1:
        terms = [] if len(fanout_terms) < 2 else fanout_terms
        terms += [] if question_count < 2 else ["multiple_question_words"]
        if terms:
            signals.append(_signal("question_fanout", terms, 0.2))

    unknown_terms = _unknown_capitalized_terms(normalized, known)
    if unknown_terms:
        signals.append(_signal("unknown_capitalized_term", unknown_terms, 0.2))

    signals.sort(key=lambda item: (item["type"], item["terms"]))
    score = min(1.0, sum(item["weight"] for item in signals) + max(0, len(signals) - 1) * 0.05)
    clarifying_terms = []
    for signal in signals:
        clarifying_terms.extend(signal["terms"])
    clarifying_terms = sorted(set(clarifying_terms), key=lambda item: (item.casefold(), item))[:max_terms_value]

    return {
        "normalized_query": normalized,
        "ambiguity_score": round(score, 3),
        "signals": signals,
        "suggested_clarifying_terms": clarifying_terms,
    }
