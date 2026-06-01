"""Audit whether uncertainty statements include reasons."""

from __future__ import annotations

import re
from typing import Any

_UNCERTAINTY = re.compile(r"\b(?:may|might|could|unclear|uncertain|unknown|not sure|appears|likely)\b", re.I)
_REASON = re.compile(r"\b(?:because|due to|since|given|depends on|varies by|evidence is limited|limited evidence)\b", re.I)


def audit_answer_uncertainty_reasoning(answer: str, evidence: Any = None, sample_limit: int = 5) -> dict[str, Any]:
    samples = []
    for index, sentence in enumerate(_sentences(answer)):
        if not _UNCERTAINTY.search(sentence):
            continue
        samples.append({"sentence_index": index, "sentence": sentence, "has_reason": bool(_REASON.search(sentence))})
    unexplained = sum(1 for sample in samples if not sample["has_reason"])
    return {
        "uncertainty_sentence_count": len(samples),
        "unexplained_uncertainty_count": unexplained,
        "has_unexplained_uncertainty": unexplained > 0,
        "samples": samples[:sample_limit],
    }


def _sentences(answer: Any) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(answer or "")) if part.strip()]
