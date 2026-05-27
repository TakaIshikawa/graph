"""Audit required terms that appear without definitions."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_DEFINITION_RE = re.compile(r"\b(?:means|refers\s+to|defined\s+as|is\s+a|is\s+an|are\s+called)\b", re.I)


def audit_answer_missing_definitions(answer: Any, required_terms: list[str]) -> list[dict[str, str]]:
    """Return required terms used without nearby definition cues."""
    if not required_terms:
        return []
    sentences = _sentences(answer)
    findings = []
    seen: set[str] = set()
    for term in required_terms:
        normalized = term.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.I)
        first = next((sentence for sentence in sentences if pattern.search(sentence)), None)
        if first is not None and not _is_defined(term, first):
            findings.append(
                {
                    "term": term,
                    "first_occurrence_sentence": first,
                    "severity": "medium",
                    "recommendation": f"Define '{term}' near its first use.",
                }
            )
    return findings


def _is_defined(term: str, sentence: str) -> bool:
    return bool(
        re.search(rf"\b{re.escape(term)}\b.{{0,80}}{_DEFINITION_RE.pattern}", sentence, re.I)
        or re.search(rf"{_DEFINITION_RE.pattern}.{{0,80}}\b{re.escape(term)}\b", sentence, re.I)
    )


def _sentences(answer: Any) -> list[str]:
    return [" ".join(match.group(0).strip().split()) for match in _SENTENCE_RE.finditer(string(answer) or "") if match.group(0).strip()]
