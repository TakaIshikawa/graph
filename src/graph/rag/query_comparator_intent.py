"""Detect comparator intent in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_INTENT_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("compare", re.compile(r"\b(?:compare|comparison|versus|vs\.?|against)\b", re.I)),
    ("rank", re.compile(r"\b(?:rank|ranking|top|bottom|best|worst)\b", re.I)),
    ("choose", re.compile(r"\b(?:choose|pick|select|which\s+(?:is|one)|between)\b", re.I)),
    ("tradeoffs", re.compile(r"\b(?:trade[-\s]?offs?|pros\s+and\s+cons|advantages|disadvantages)\b", re.I)),
    ("evaluate", re.compile(r"\b(?:evaluate|assess|score|benchmark)\b", re.I)),
)
_VS_RE = re.compile(r"\b([A-Za-z0-9][\w .+-]{1,50}?)\s+(?:vs\.?|versus)\s+([A-Za-z0-9][\w .+-]{1,50}?)(?:[?.!,;]|$)", re.I)
_BETWEEN_RE = re.compile(r"\bbetween\s+([A-Za-z0-9][\w .+-]{1,50}?)\s+and\s+([A-Za-z0-9][\w .+-]{1,50}?)(?:[?.!,;]|$)", re.I)


def detect_query_comparator_intent(query: str) -> dict[str, Any]:
    normalized = _normalize_query(query)
    matches = [
        {"intent": label, "matched_text": match.group(0)}
        for label, pattern in _INTENT_SPECS
        if (match := pattern.search(normalized))
    ]
    labels = sorted({row["intent"] for row in matches})
    entities = _compared_entities(normalized)
    confidence = _confidence(labels, entities)
    return {
        "has_comparator_intent": bool(labels),
        "intent_labels": labels,
        "matched_terms": sorted(matches, key=lambda row: (row["intent"], row["matched_text"].casefold())),
        "compared_entities": entities,
        "confidence": confidence,
        "normalized_query": normalized,
    }


def _compared_entities(query: str) -> list[str]:
    for pattern in (_VS_RE, _BETWEEN_RE):
        match = pattern.search(query)
        if match:
            return [_clean_entity(match.group(1)), _clean_entity(match.group(2))]
    return []


def _clean_entity(value: str) -> str:
    return re.sub(r"^(?:compare|choose|pick|select)\s+", "", value.strip(), flags=re.I)


def _confidence(labels: list[str], entities: list[str]) -> float:
    if not labels:
        return 0.0
    return round(min(0.95, 0.55 + 0.1 * len(labels) + (0.15 if entities else 0.0)), 2)


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
