"""Audit numeric range claims against evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import string, value

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_NUM = r"\d+(?:\.\d+)?(?:x|%)?"
_RANGE_PATTERNS = (
    re.compile(rf"\b({_NUM})\s*[-–]\s*({_NUM})(?=\W|$)", re.I),
    re.compile(rf"\bbetween\s+({_NUM})\s+and\s+({_NUM})(?=\W|$)", re.I),
    re.compile(rf"\bfrom\s+({_NUM})\s+to\s+({_NUM})(?=\W|$)", re.I),
)


def audit_answer_range_claims(answer: str, evidence: Iterable[Any]) -> list[dict[str, Any]]:
    evidence_texts = [_evidence_text(item).casefold() for item in evidence or []]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _sentences(answer):
        for start, end in _ranges(sentence):
            normalized = f"{start} to {end}"
            key = (sentence.casefold(), normalized.casefold())
            if key in seen:
                continue
            seen.add(key)
            match_count = sum(1 for text in evidence_texts if start.casefold() in text and end.casefold() in text)
            rows.append({"claim_text": sentence, "normalized_range": normalized, "evidence_match_count": match_count, "severity": "none" if match_count else "medium"})
    return rows


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(text or "")) if match.group(0).strip()]


def _ranges(sentence: str) -> list[tuple[str, str]]:
    found = []
    for pattern in _RANGE_PATTERNS:
        found.extend((match.group(1), match.group(2)) for match in pattern.finditer(sentence))
    return found


def _evidence_text(item: Any) -> str:
    return " ".join(text for key in ("text", "snippet", "content") if (text := string(value(item, key))))
