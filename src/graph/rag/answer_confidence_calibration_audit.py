"""Audit answer confidence wording for calibration risks."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import string

_CONFIDENCE_RE = re.compile(r"(?i)\b(?:\d{1,3}\s*%\s+confidence|confidence\s+(?:level\s+)?(?:is\s+)?\d{1,3}\s*%|high confidence|low confidence|very confident|confident)\b")
_UNCERTAINTY_RE = re.compile(r"(?i)\b(?:may|might|could|likely|unlikely|appears|seems|suggests|unclear|uncertain|not enough evidence|limited evidence|preliminary)\b")
_CERTAINTY_RE = re.compile(r"(?i)\b(?:always|never|guaranteed|certainly|definitely|undeniably|without doubt|no question|proves?|must|all|none)\b")
_ABSOLUTE_PHRASE_RE = re.compile(r"(?i)\b(?:the only possible|the exact answer|cannot be wrong|settled fact|everyone agrees|there is no evidence against)\b")
_EVIDENCE_RE = re.compile(r"(?i)\b(?:according to|source|citation|cited|evidence|study|data|reported|measured)\b|\[[^\]]+\]|\(\d{4}\)")


def audit_answer_confidence_calibration(answer: str) -> dict[str, Any]:
    """Return confidence calibration signals for an answer."""
    text = string(answer) or ""
    if not text:
        return {
            "calibration_score": 0.0,
            "confidence_claim_count": 0,
            "uncertainty_marker_count": 0,
            "overconfidence_flags": [],
            "numeric_confidence_percentages": [],
            "recommendation": "Add evidence-qualified confidence language before making claims.",
        }

    confidence_claims = [match.group(0) for match in _CONFIDENCE_RE.finditer(text)]
    uncertainty_markers = [match.group(0).casefold() for match in _UNCERTAINTY_RE.finditer(text)]
    numeric_percentages = [int(match.group(1)) for match in re.finditer(r"(?i)\b(\d{1,3})\s*%\s+confidence\b|\bconfidence\s+(?:level\s+)?(?:is\s+)?(\d{1,3})\s*%", text) for group in match.groups() if group is not None]
    overconfidence_flags = []
    has_evidence = bool(_EVIDENCE_RE.search(text))

    for match in _CERTAINTY_RE.finditer(text):
        flag = {"phrase": match.group(0), "start": match.start(), "kind": "unsupported_certainty"}
        if has_evidence:
            flag["kind"] = "absolute_language"
        overconfidence_flags.append(flag)
    for match in _ABSOLUTE_PHRASE_RE.finditer(text):
        overconfidence_flags.append({"phrase": match.group(0), "start": match.start(), "kind": "overconfident_absolute_phrase"})

    score = 0.45 + min(len(uncertainty_markers), 4) * 0.12 + min(len(confidence_claims), 2) * 0.05
    score -= len([flag for flag in overconfidence_flags if flag["kind"] != "absolute_language"]) * 0.18
    score -= len([flag for flag in overconfidence_flags if flag["kind"] == "absolute_language"]) * 0.08
    score = round(max(0.0, min(1.0, score)), 2)

    if overconfidence_flags and not uncertainty_markers:
        recommendation = "Temper absolute claims and tie confidence to cited evidence."
    elif uncertainty_markers:
        recommendation = "Confidence language is better calibrated; keep uncertainty tied to evidence."
    else:
        recommendation = "Add explicit confidence or uncertainty markers where evidence is limited."

    return {
        "calibration_score": score,
        "confidence_claim_count": len(confidence_claims),
        "uncertainty_marker_count": len(uncertainty_markers),
        "overconfidence_flags": overconfidence_flags,
        "numeric_confidence_percentages": numeric_percentages,
        "recommendation": recommendation,
    }
