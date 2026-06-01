"""Flag relative answer dates that lack absolute anchors."""

from __future__ import annotations

import re
from typing import Any

_PHRASES = ("today", "yesterday", "tomorrow", "last year", "next quarter", "recently", "currently", "soon")
_PHRASE_RE = re.compile(r"\b(" + "|".join(re.escape(phrase) for phrase in _PHRASES) + r")\b", re.I)
_ABSOLUTE_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b",
    re.I,
)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def audit_answer_date_ambiguity(answer: str) -> list[dict[str, Any]]:
    rows = []
    for sentence in _sentences(answer):
        if _ABSOLUTE_DATE_RE.search(sentence):
            continue
        for match in _PHRASE_RE.finditer(sentence):
            rows.append(
                {
                    "phrase": match.group(1).casefold(),
                    "sentence": sentence,
                    "severity": "medium",
                    "suggestion": "Replace the relative date with an absolute date.",
                }
            )
    return rows


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(text or "")) if match.group(0).strip()]
