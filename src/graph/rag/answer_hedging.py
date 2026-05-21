"""Audit answer text for hedging and unsupported certainty cues."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_CERTAINTY_RE = re.compile(r"(?i)\b(?:always|never|guaranteed|proves?|undeniable|certainly|definitely|must|all|none)\b")
_HEDGE_RE = re.compile(r"(?i)\b(?:maybe|possibly|unclear|might|could|may|appears|seems|likely|perhaps|suggests)\b")
_EVIDENCE_RE = re.compile(r"(?i)\b(?:according to|cited|citation|source|evidence|study|data|reported|measured)\b")


def _sentences(answer: str) -> list[tuple[int, str]]:
    return [(match.start(), match.group(0).strip()) for match in _SENTENCE_RE.finditer(answer) if match.group(0).strip()]


def _severity(kind: str, cue_count: int, has_evidence: bool) -> str:
    if kind == "unsupported_certainty":
        if has_evidence:
            return "low"
        return "high" if cue_count > 1 else "medium"
    if cue_count >= 3:
        return "high"
    if cue_count == 2:
        return "medium"
    return "low"


def _balance(counts: Counter[str]) -> str:
    certainty = counts["unsupported_certainty"]
    hedge = counts["excessive_uncertainty"]
    evidence = counts["evidence_reference"]
    if certainty == hedge == 0:
        return "neutral"
    if certainty > hedge and certainty > evidence:
        return "overconfident"
    if hedge >= certainty + 2:
        return "over_hedged"
    if evidence:
        return "evidence_balanced"
    return "mixed"


def audit_answer_hedging(answer: str) -> dict[str, Any]:
    """Return sentence-level hedging records and an aggregate balance bucket."""
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, (start, sentence) in enumerate(_sentences(answer)):
        evidence_cues = [match.group(0).casefold() for match in _EVIDENCE_RE.finditer(sentence)]
        certainty_cues = [match.group(0).casefold() for match in _CERTAINTY_RE.finditer(sentence)]
        hedge_cues = [match.group(0).casefold() for match in _HEDGE_RE.finditer(sentence)]
        has_evidence = bool(evidence_cues)
        if evidence_cues:
            counts["evidence_reference"] += 1
            records.append(
                {
                    "sentence_index": index,
                    "start": start,
                    "kind": "evidence_reference",
                    "cue": evidence_cues[0],
                    "severity": "low",
                    "sentence": sentence,
                }
            )
        if certainty_cues:
            counts["unsupported_certainty"] += 1
            records.append(
                {
                    "sentence_index": index,
                    "start": start,
                    "kind": "unsupported_certainty",
                    "cue": certainty_cues[0],
                    "severity": _severity("unsupported_certainty", len(certainty_cues), has_evidence),
                    "sentence": sentence,
                }
            )
        if hedge_cues and (len(hedge_cues) >= 2 or len(hedge_cues) / max(len(sentence.split()), 1) > 0.08):
            counts["excessive_uncertainty"] += 1
            records.append(
                {
                    "sentence_index": index,
                    "start": start,
                    "kind": "excessive_uncertainty",
                    "cue": hedge_cues[0],
                    "severity": _severity("excessive_uncertainty", len(hedge_cues), has_evidence),
                    "sentence": sentence,
                }
            )
    return {
        "sentence_count": len(_sentences(answer)),
        "records": records,
        "counts": {
            "unsupported_certainty": counts["unsupported_certainty"],
            "excessive_uncertainty": counts["excessive_uncertainty"],
            "evidence_reference": counts["evidence_reference"],
        },
        "balance_bucket": _balance(counts),
    }
