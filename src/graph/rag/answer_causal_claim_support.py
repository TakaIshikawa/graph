"""Audit causal claims in answers against evidence method signals."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, metadata

_CAUSE_RE = re.compile(r"\b(causes?|caused|because|leads? to|drives?|results? in|due to|impact(?:s|ed)?)\b", re.I)
_STRONG_RE = re.compile(r"\b(randomi[sz]ed|controlled|longitudinal|mechanism|experiment(?:al)?)\b", re.I)
_WEAK_RE = re.compile(r"\b(observational|correlation(?:al)?|associated|survey)\b", re.I)


def audit_answer_causal_claim_support(answer: str, evidence: Iterable[Any]) -> dict[str, Any]:
    claims = [
        {"sentence": sentence, "cue_words": _dedupe(m.group(1).casefold() for m in _CAUSE_RE.finditer(sentence))}
        for sentence in _sentences(answer)
        if _CAUSE_RE.search(sentence)
    ]
    strong = weak = 0
    for item in evidence:
        text = content_text(item) + " " + " ".join(str(v) for v in metadata(item).values())
        strong += int(bool(_STRONG_RE.search(text)))
        weak += int(bool(_WEAK_RE.search(text)))
    unsupported = [] if strong else claims
    return {
        "causal_claims": claims,
        "unsupported_causal_claims": unsupported,
        "support_summary": {"causal_claim_count": len(claims), "strong_support_count": strong, "correlational_only_count": weak},
    }


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
