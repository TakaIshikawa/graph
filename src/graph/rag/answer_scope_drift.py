"""Audit answer scope drift against query constraints."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_CUE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "exclusion": (
        re.compile(r"\b(?:exclude|excluding|without|avoid|not\s+(?:including|about)|except)\s+([^.;,]+)", re.I),
    ),
    "geographic": (
        re.compile(r"\b(?:in|for|within|across)\s+(?:the\s+)?(US|U\.S\.|United States|EU|Europe|UK|Japan|Canada|California|Texas|New York)\b", re.I),
        re.compile(r"\b(?:region|country|jurisdiction|state):\s*([^.;,]+)", re.I),
    ),
    "temporal": (
        re.compile(r"\b(?:before|after|since|until|through)\s+((?:19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?)\b", re.I),
        re.compile(r"\b(?:last|past)\s+\d+\s+(?:days|weeks|months|years)\b", re.I),
        re.compile(r"\b(?:only|just)\s+(?:in|for)\s+((?:19|20)\d{2})\b", re.I),
    ),
    "source": (
        re.compile(r"\b(?:from|using|cite|include)\s+(?:only\s+)?(?:official|peer-reviewed|primary|government|academic|company)\s+(?:sources|reports|data|evidence)\b", re.I),
        re.compile(r"\b(?:only|just)\s+(?:official|peer-reviewed|primary|government|academic|company)\b", re.I),
    ),
}

_ANSWER_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "exclusion": (re.compile(r"\b(?:also|including|includes|covers|discusses)\s+([^.;]+)", re.I),),
    "geographic": (re.compile(r"\b(?:worldwide|globally|internationally|in\s+all\s+regions|across\s+countries|everywhere)\b", re.I),),
    "temporal": (re.compile(r"\b(?:historically|over\s+time|always|in\s+all\s+periods|from\s+all\s+years|current\s+and\s+past)\b", re.I),),
    "source": (re.compile(r"\b(?:generally|sources\s+vary|blogs?|forums?|social\s+media|any\s+source|all\s+sources)\b", re.I),),
}

_BROAD_PATTERNS = (
    re.compile(r"\b(?:generally|everyone|all\s+cases|in\s+all\s+cases|always|never|universally)\b", re.I),
)


def audit_answer_scope_drift(query: Any, answer: Any) -> dict[str, Any]:
    """Return likely scope drift where an answer broadens constrained queries."""
    query_text = string(query) or ""
    answer_text = string(answer) or ""
    query_cues = _query_cues(query_text)
    answer_cues = _answer_cues(answer_text)
    matched = []

    for query_cue in query_cues:
        for answer_cue in answer_cues:
            if answer_cue["type"] == query_cue["type"] or answer_cue["type"] == "broad":
                matched.append({"drift_type": query_cue["type"], "query_cue": query_cue["cue"], "answer_cue": answer_cue["cue"]})

    drift_types = sorted({item["drift_type"] for item in matched})
    return {
        "has_scope_drift_risk": bool(matched),
        "drift_types": drift_types,
        "matched_cues": matched,
    }


def _query_cues(text: str) -> list[dict[str, str]]:
    cues = []
    for drift_type, patterns in _CUE_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                cues.append({"type": drift_type, "cue": match.group(0)})
    if re.search(r"\b(?:only|just|limited\s+to|for)\b", text, re.I) and not cues:
        cues.append({"type": "general", "cue": "limited scope"})
    return cues


def _answer_cues(text: str) -> list[dict[str, str]]:
    cues = []
    for drift_type, patterns in _ANSWER_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                cues.append({"type": drift_type, "cue": match.group(0)})
    for pattern in _BROAD_PATTERNS:
        for match in pattern.finditer(text):
            cues.append({"type": "broad", "cue": match.group(0)})
    return cues
