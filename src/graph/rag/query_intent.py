"""Deterministic lexical intent classification for RAG queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _IntentCue:
    intent: str
    cue: str
    weight: float
    pattern: re.Pattern[str]


def _cue_pattern(*terms: str) -> re.Pattern[str]:
    alternatives = [re.escape(term).replace(r"\ ", r"\s+") for term in terms]
    return re.compile(
        rf"(?<![A-Za-z0-9])(?:{'|'.join(alternatives)})(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _regex_cue(intent: str, cue: str, weight: float, pattern: str) -> _IntentCue:
    return _IntentCue(intent, cue, weight, re.compile(pattern, re.IGNORECASE))


_INTENT_PRIORITY = {
    "contradiction_check": 0,
    "comparison": 1,
    "timeline": 2,
    "how_to": 3,
    "definition": 4,
    "lookup": 5,
    "exploratory": 6,
}

_SEARCH_MODES = {
    "lookup": "full_text",
    "comparison": "hybrid",
    "timeline": "hybrid",
    "how_to": "semantic",
    "definition": "semantic",
    "contradiction_check": "hybrid",
    "exploratory": "semantic",
}

_INTENT_CUES = (
    _IntentCue(
        "contradiction_check",
        "contradicts",
        3.0,
        _cue_pattern("contradict", "contradicts", "contradicted", "contradiction"),
    ),
    _IntentCue(
        "contradiction_check",
        "conflicts",
        3.0,
        _cue_pattern("conflict", "conflicts", "conflicting"),
    ),
    _IntentCue(
        "contradiction_check",
        "disagrees",
        2.8,
        _cue_pattern("disagree", "disagrees", "disagreed", "dispute", "disputed"),
    ),
    _IntentCue(
        "contradiction_check",
        "inconsistent",
        2.8,
        _cue_pattern("inconsistent", "inconsistency"),
    ),
    _IntentCue(
        "contradiction_check",
        "retracted",
        2.6,
        _cue_pattern("retracted", "retraction", "withdrawn"),
    ),
    _IntentCue(
        "contradiction_check",
        "debunked",
        2.4,
        _cue_pattern("debunk", "debunked", "refute", "refuted"),
    ),
    _IntentCue("comparison", "compare", 2.7, _cue_pattern("compare", "comparison", "contrast")),
    _IntentCue("comparison", "versus", 2.7, _cue_pattern("versus", "vs")),
    _IntentCue(
        "comparison",
        "difference between",
        2.5,
        _cue_pattern("difference between", "differences between"),
    ),
    _IntentCue(
        "comparison",
        "tradeoff",
        2.3,
        _cue_pattern("tradeoff", "tradeoffs", "trade-off", "trade-offs"),
    ),
    _IntentCue("comparison", "better than", 2.1, _cue_pattern("better than", "worse than")),
    _IntentCue("timeline", "timeline", 2.8, _cue_pattern("timeline", "chronology", "history")),
    _IntentCue("timeline", "when", 2.2, _cue_pattern("when", "date", "dates")),
    _IntentCue("timeline", "before", 2.1, _cue_pattern("before", "after", "since", "during")),
    _IntentCue(
        "timeline",
        "recent",
        1.8,
        _cue_pattern("recent", "latest", "newest", "last year", "this year"),
    ),
    _IntentCue("how_to", "how to", 3.0, _cue_pattern("how to", "how do i", "how can i")),
    _IntentCue("how_to", "steps", 2.4, _cue_pattern("steps", "step by step", "instructions")),
    _IntentCue("how_to", "guide", 2.1, _cue_pattern("guide", "tutorial", "workflow", "playbook")),
    _IntentCue("definition", "what is", 3.0, _cue_pattern("what is", "what are", "what does")),
    _IntentCue("definition", "define", 2.8, _cue_pattern("define", "definition", "meaning of")),
    _IntentCue("definition", "explain", 2.0, _cue_pattern("explain", "overview of", "summary of")),
    _IntentCue("lookup", "find", 2.4, _cue_pattern("find", "search for", "lookup", "look up")),
    _IntentCue("lookup", "show", 2.1, _cue_pattern("show", "list", "give me")),
    _IntentCue("lookup", "who", 2.0, _cue_pattern("who", "where", "which")),
    _IntentCue("lookup", "exact", 1.8, _cue_pattern("exact", "specific", "named")),
    _IntentCue("exploratory", "explore", 2.2, _cue_pattern("explore", "browse", "survey")),
    _IntentCue("exploratory", "related", 2.0, _cue_pattern("related", "similar", "adjacent")),
    _IntentCue(
        "exploratory",
        "themes",
        1.9,
        _cue_pattern("themes", "ideas", "patterns", "topics"),
    ),
    _regex_cue(
        "timeline",
        "year",
        1.8,
        r"(?<![A-Za-z0-9])(?:19|20)\d{2}(?![A-Za-z0-9])",
    ),
)

_DATE_RELATIVE_PATTERN = _cue_pattern(
    "recent",
    "latest",
    "newest",
    "today",
    "yesterday",
    "last year",
    "this year",
)
_YEAR_PATTERN = re.compile(r"(?<![A-Za-z0-9])((?:19|20)\d{2})(?![A-Za-z0-9])")
_BEFORE_YEAR_PATTERN = re.compile(
    r"\b(?:before|until|through)\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_AFTER_YEAR_PATTERN = re.compile(r"\b(?:after|since|from)\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_TAG_PATTERN = re.compile(r"(?<!\w)(?:tag:|#)([A-Za-z0-9][A-Za-z0-9_.-]*)")
_SOURCE_PATTERN = re.compile(
    r"\b(?:source|from)\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9_.-]*)",
    re.IGNORECASE,
)
_SOURCE_STOPWORDS = frozenset(
    {"before", "after", "since", "during", "tag", "tags", "source", "sources"}
)


def _normalize_query(query: str) -> str:
    return " ".join(str(query).split())


def _matched_intent_cues(query: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores = {intent: 0.0 for intent in _INTENT_PRIORITY}
    cues_by_intent = {intent: [] for intent in _INTENT_PRIORITY}
    seen_by_intent = {intent: set() for intent in _INTENT_PRIORITY}

    for cue in _INTENT_CUES:
        if not cue.pattern.search(query):
            continue
        scores[cue.intent] += cue.weight
        if cue.cue not in seen_by_intent[cue.intent]:
            cues_by_intent[cue.intent].append(cue.cue)
            seen_by_intent[cue.intent].add(cue.cue)

    return scores, cues_by_intent


def _choose_intent(scores: dict[str, float]) -> str:
    best_intent = "exploratory"
    best_score = 0.0
    for intent in _INTENT_PRIORITY:
        score = scores[intent]
        if score > best_score:
            best_intent = intent
            best_score = score
    return best_intent


def _confidence(score: float, matched_cues: list[str]) -> float:
    if score <= 0:
        return 0.2
    value = 0.45 + min(score, 5.0) / 10 + min(len(matched_cues), 3) * 0.05
    return round(min(value, 0.95), 2)


def _metadata_filters(query: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    tags = sorted({match.group(1).lower() for match in _TAG_PATTERN.finditer(query)})
    if tags:
        filters["tags"] = tags

    sources = sorted(
        {
            match.group(1).lower()
            for match in _SOURCE_PATTERN.finditer(query)
            if match.group(1).lower() not in _SOURCE_STOPWORDS
            and not _YEAR_PATTERN.fullmatch(match.group(1))
        }
    )
    if sources:
        filters["source_project"] = sources

    date_filter: dict[str, Any] = {}
    years = sorted({match.group(1) for match in _YEAR_PATTERN.finditer(query)})
    if years:
        date_filter["years"] = years
    before_years = sorted({match.group(1) for match in _BEFORE_YEAR_PATTERN.finditer(query)})
    if before_years:
        date_filter["before"] = before_years[0]
    after_years = sorted({match.group(1) for match in _AFTER_YEAR_PATTERN.finditer(query)})
    if after_years:
        date_filter["after"] = after_years[-1]
    if _DATE_RELATIVE_PATTERN.search(query):
        date_filter["relative"] = "recent"
    if date_filter:
        filters["date"] = date_filter

    return filters


def classify_query_intent(query: str) -> dict[str, Any]:
    """Classify a user search query into a deterministic retrieval intent.

    The classifier uses lexical cues only. Suggested search modes are stable:
    exact lookups use ``full_text``, broad meaning-oriented requests use
    ``semantic``, and requests that usually need both cue matching and semantic
    expansion use ``hybrid``.
    """
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return {
            "intent": "exploratory",
            "confidence": 0.1,
            "matched_cues": [],
            "suggested_search_mode": "semantic",
            "suggested_metadata_filters": {},
        }

    scores, cues_by_intent = _matched_intent_cues(normalized_query)
    intent = _choose_intent(scores)
    matched_cues = cues_by_intent[intent]

    return {
        "intent": intent,
        "confidence": _confidence(scores[intent], matched_cues),
        "matched_cues": matched_cues,
        "suggested_search_mode": _SEARCH_MODES[intent],
        "suggested_metadata_filters": _metadata_filters(normalized_query),
    }
