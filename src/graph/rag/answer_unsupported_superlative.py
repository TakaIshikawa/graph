"""Audit answer superlatives that lack nearby citation markers."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_SUPERLATIVE_RE = re.compile(r"(?i)\b(strongest|best|first|only|largest|fastest|highest|lowest|most|least|newest|oldest)\b")
_CITATION_RE = re.compile(r"\[[^\]]+\]|\(\d{4}\)|(?i:\baccording to\b|\bsource\b|\bcited\b|\bevidence\b)")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")


def audit_answer_unsupported_superlatives(answer: str) -> dict[str, Any]:
    text = string(answer) or ""
    flagged: list[dict[str, Any]] = []
    cited: list[dict[str, Any]] = []

    for sentence_index, match in enumerate(_SENTENCE_RE.finditer(text)):
        sentence = match.group(0).strip()
        if _is_quote(sentence):
            continue
        for superlative in _SUPERLATIVE_RE.finditer(sentence):
            start = match.start() + superlative.start()
            record = {
                "sentence_index": sentence_index,
                "phrase": superlative.group(0).casefold(),
                "sentence": sentence,
                "has_nearby_citation": _has_nearby_citation(text, start, match.start(), match.end()),
            }
            if record["has_nearby_citation"]:
                cited.append(record)
            else:
                flagged.append(record)

    return {"flagged_claims": flagged, "cited_superlatives": cited, "unsupported_count": len(flagged), "cited_count": len(cited)}


def _has_nearby_citation(text: str, start: int, sentence_start: int, sentence_end: int) -> bool:
    window = text[max(0, start - 80) : min(len(text), start + 120)]
    return bool(_CITATION_RE.search(window) or _CITATION_RE.search(text[sentence_start:sentence_end]))


def _is_quote(sentence: str) -> bool:
    stripped = sentence.strip()
    return stripped.startswith(("\"", "'", ">")) or stripped.startswith(("“", "‘"))
