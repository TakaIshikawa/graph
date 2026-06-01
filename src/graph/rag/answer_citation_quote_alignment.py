"""Audit whether quoted answer phrases appear in retrieved evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import string, value

_QUOTE_RE = re.compile(r'"([^"\n]{3,120})"|\'([^\'\n]{3,120})\'|“([^”\n]{3,120})”|‘([^’\n]{3,120})’')
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def audit_answer_citation_quote_alignment(answer: str, evidence: Iterable[Any]) -> list[dict[str, Any]]:
    """Return one row per unique short quote in the answer."""
    evidence_texts = [_evidence_text(item).casefold() for item in evidence or []]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sentence in _sentences(answer):
        for quote in _quotes(sentence):
            key = quote.casefold()
            if key in seen:
                continue
            seen.add(key)
            match_count = sum(1 for text in evidence_texts if key in text)
            rows.append(
                {
                    "quote": quote,
                    "evidence_match_count": match_count,
                    "severity": "none" if match_count else "medium",
                    "claim_text": sentence,
                }
            )
    return rows


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(text or "")) if match.group(0).strip()]


def _quotes(text: str) -> list[str]:
    return [next(group for group in match.groups() if group is not None).strip() for match in _QUOTE_RE.finditer(text)]


def _evidence_text(item: Any) -> str:
    parts = []
    for key in ("text", "snippet", "content"):
        text = string(value(item, key))
        if text:
            parts.append(text)
    return " ".join(parts)
