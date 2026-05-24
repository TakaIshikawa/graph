"""Audit answer sentences that drift beyond the original query scope."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import tokens

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*\b")
_TASK_TERMS = {"compare", "rank", "summarize", "explain", "recommend", "calculate", "list", "audit"}
_GEO_TERMS = {"us", "usa", "europe", "asia", "africa", "japan", "china", "canada", "uk", "india", "brazil", "global"}


def audit_answer_scope_creep(query: str, answer: str) -> dict[str, Any]:
    """Flag answer spans that introduce unrelated entities, geographies, dates, or intents."""
    scope = _scope(query)
    findings = []
    for sentence in _sentences(answer):
        sentence_scope = _scope(sentence)
        reasons = []
        if sentence_scope["entities"] - scope["entities"]:
            reasons.append("new_entity")
        if sentence_scope["years"] - scope["years"] and scope["years"]:
            reasons.append("new_date")
        if sentence_scope["geographies"] - scope["geographies"] and scope["geographies"]:
            reasons.append("new_geography")
        if sentence_scope["tasks"] - scope["tasks"] and scope["tasks"]:
            reasons.append("new_task_intent")
        if reasons:
            findings.append({"span_text": sentence, "reason_codes": reasons, "severity": "medium" if len(reasons) == 1 else "high"})
    return {"scope_terms": sorted(scope["terms"]), "findings": findings}


def _scope(text: str) -> dict[str, set[str]]:
    normalized = str(text or "")
    terms = tokens(normalized, min_length=4)
    return {
        "terms": terms,
        "entities": _entities(normalized),
        "years": set(_YEAR_RE.findall(normalized)),
        "tasks": terms & _TASK_TERMS,
        "geographies": {term for term in tokens(normalized, min_length=2) if term in _GEO_TERMS},
    }


def _entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        words = [word.casefold() for word in match.group(0).split()]
        entities.update(word for word in words if word in _GEO_TERMS)
        if len(words) > 1 and words[0] not in _TASK_TERMS:
            entities.add(" ".join(words))
    return entities


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(text or "")) if match.group(0).strip()]
