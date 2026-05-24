"""Label source mentions in answer text by likely evidence role."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_BRACKET_RE = re.compile(r"\[(?:\d+|[A-Za-z][\w.-]*(?:,\s*\d{4})?)\]")
_URL_RE = re.compile(r"https?://[^\s)]+")
_NAMED_RE = re.compile(r"\b(?i:(?:according to|from|in|the))\s+([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4})")
_ROLE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("counterexample", re.compile(r"\b(?:however|but|counterexample|contradict|conflict|unlike)\b", re.I)),
    ("methodology", re.compile(r"\b(?:method|sample|survey|dataset|measured|participants|n\s*=)\b", re.I)),
    ("definition", re.compile(r"\b(?:defines?|definition|means|refers to|is defined as)\b", re.I)),
    ("background_context", re.compile(r"\b(?:background|context|overview|history|explains)\b", re.I)),
    ("primary_evidence", re.compile(r"\b(?:shows|found|reports?|according to|evidence|study|trial)\b", re.I)),
)


def label_answer_source_roles(answer: str) -> dict[str, Any]:
    """Return source mentions and aggregate role counts inferred from sentence cues."""
    mentions = []
    for sentence in _sentences(answer):
        role = _role(sentence)
        for source in _sources(sentence):
            mentions.append({"source": source, "role": role, "sentence": sentence})
    counts = Counter(mention["role"] for mention in mentions)
    return {"source_mentions": mentions, "role_counts": dict(sorted(counts.items()))}


def _sentences(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split())
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]


def _sources(sentence: str) -> list[str]:
    sources: list[str] = []
    for pattern in (_BRACKET_RE, _URL_RE):
        sources.extend(match.group(0).rstrip(".,") for match in pattern.finditer(sentence))
    for match in _NAMED_RE.finditer(sentence):
        source = match.group(1).strip().rstrip(".,")
        if source.casefold() not in {"the", "a", "an"}:
            sources.append(source)
    seen = set()
    unique = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


def _role(sentence: str) -> str:
    for role, pattern in _ROLE_CUES:
        if pattern.search(sentence):
            return role
    return "background_context"
