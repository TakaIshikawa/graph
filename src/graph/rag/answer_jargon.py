"""Audit answers for unexplained jargon and acronym density."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")
_TECH_TERMS = ("embedding", "vector", "retrieval", "inference", "latency", "reranking", "tokenization", "ontology", "schema")


def audit_answer_jargon(answer: str, *, allowed_terms: Iterable[str] | None = None) -> dict[str, Any]:
    text = str(answer or "")
    allowed = {term.casefold() for term in (allowed_terms or [])}
    explained = _explained_acronyms(text)
    counts = Counter(match.group(0) for match in _ACRONYM_RE.finditer(text))
    flagged: list[str] = []
    for term in sorted(counts, key=lambda value: value.casefold()):
        if term.casefold() not in allowed and term not in explained and counts[term] >= 2:
            flagged.append(term)
    lower = text.casefold()
    for term in _TECH_TERMS:
        if term not in allowed and lower.count(term) >= 2 and not re.search(rf"\b{re.escape(term)}\b\s+(?:means|refers\s+to|is\s+a|is\s+an)", text, re.I):
            flagged.append(term)
    flagged = sorted(dict.fromkeys(flagged), key=lambda value: value.casefold())
    word_count = len(re.findall(r"\b\w+\b", text))
    return {
        "word_count": word_count,
        "jargon_count": len(flagged),
        "flagged_terms": flagged,
        "passes": not flagged,
    }


def _explained_acronyms(text: str) -> set[str]:
    explained = set()
    for match in re.finditer(r"\b[A-Za-z][A-Za-z\s-]{3,80}\s+\(([A-Z]{2,})\)", text):
        explained.add(match.group(1))
    for match in re.finditer(r"\b([A-Z]{2,})\s+\([A-Za-z][A-Za-z\s-]{3,80}\)", text):
        explained.add(match.group(1))
    return explained
