"""Audit strong answer claims that lack citation markers."""

from __future__ import annotations

import re
from typing import Any

_CUES = ("always", "never", "proven", "guarantees", "all", "none", "best", "only")
_CITATION_RE = re.compile(r"\[\d+\]|\([A-Z][A-Za-z-]+,\s*\d{4}\)|https?://\S+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def audit_answer_citation_overclaims(answer: str, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(answer or "")) if match.group(0).strip()]
    strong_claim_count = uncited = 0
    samples: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences, start=1):
        cue = next((cue for cue in _CUES if re.search(rf"\b{re.escape(cue)}\b", sentence, re.I)), None)
        if cue is None:
            continue
        strong_claim_count += 1
        has_citation = bool(_CITATION_RE.search(sentence))
        if not has_citation:
            uncited += 1
        if len(samples) < limit:
            samples.append({"sentence_index": index, "cue": cue, "sentence": sentence, "has_citation": has_citation})
    return {
        "sentence_count": len(sentences),
        "strong_claim_count": strong_claim_count,
        "uncited_strong_claim_count": uncited,
        "samples": samples,
    }
