"""Audit drafted RAG answers for common assumption cues."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string, tokens

_PATTERNS = [
    ("unstated_causality", "medium", re.compile(r"\b(because|therefore|so|caused by|leads to|drives|results in)\b", re.I), "verify causal evidence"),
    ("inferred_intent", "medium", re.compile(r"\b(wants to|intends to|is trying to|goal is|likely wants)\b", re.I), "confirm user or actor intent"),
    ("generalized_scope", "medium", re.compile(r"\b(all|always|never|everyone|no one|only|any|must)\b", re.I), "check scope and exceptions"),
    ("unsupported_recommendation", "medium", re.compile(r"\b(should|recommend|best option|need to|must choose|ought to)\b", re.I), "verify recommendation criteria"),
    ("hedged_uncertainty", "low", re.compile(r"\b(may|might|could|possibly|appears|seems|likely|unclear)\b", re.I), "clarify uncertainty with evidence"),
]


def audit_answer_assumptions(answer: Any, evidence: Any = None) -> dict[str, Any]:
    """Return assumption records with stable codes and severity."""
    text = string(answer) or ""
    evidence_terms = tokens(evidence, min_length=4)
    records = []
    for code, severity, pattern, suggested_check in _PATTERNS:
        for match in pattern.finditer(text):
            span = _sentence(text, match.start(), match.end())
            supported = bool(evidence_terms and (tokens(span, min_length=4) & evidence_terms))
            records.append(
                {
                    "code": code,
                    "severity": _reduced(severity) if supported else severity,
                    "span": span[:160],
                    "reason": f"{code}_cue",
                    "suggested_check": suggested_check,
                    "support_signal": "evidence_overlap" if supported else None,
                }
            )
            break
    warnings = []
    if any(record["severity"] == "high" for record in records):
        warnings.append("high_severity_assumptions")
    if records:
        warnings.append("assumptions_detected")
    return {"assumption_count": len(records), "assumptions": records, "warnings": warnings}


def _sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    right_dot = text.find(".", end)
    right_line = text.find("\n", end)
    candidates = [pos for pos in (right_dot, right_line) if pos != -1]
    right = min(candidates) + 1 if candidates else len(text)
    return " ".join(text[left:right].strip().split())


def _reduced(severity: str) -> str:
    return {"high": "medium", "medium": "low"}.get(severity, severity)
