"""Audit broad quantifier phrases in RAG answers."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_QUANTIFIERS = {
    "all": "high",
    "always": "high",
    "never": "high",
    "none": "high",
    "every": "high",
    "majority": "medium",
    "most": "medium",
    "many": "low",
}


def audit_answer_claim_quantifiers(answer: Any) -> list[dict[str, str]]:
    """Return deterministic findings for broad quantifier language."""
    findings = []
    for sentence in _sentences(answer):
        for match in re.finditer(r"\b(all|always|never|none|every|majority|most|many)\b", sentence, re.I):
            normalized = match.group(1).casefold()
            findings.append(
                {
                    "phrase": match.group(0),
                    "normalized_quantifier": normalized,
                    "sentence": sentence,
                    "severity": _QUANTIFIERS[normalized],
                    "recommendation": "Qualify broad quantifiers or cite evidence that supports the scope.",
                }
            )
    return findings


def _sentences(answer: Any) -> list[str]:
    return [" ".join(match.group(0).strip().split()) for match in _SENTENCE_RE.finditer(string(answer) or "") if match.group(0).strip()]
