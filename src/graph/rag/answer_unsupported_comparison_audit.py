"""Audit comparison claims in answers against available evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, string

_CUES = ("better than", "worse than", "more than", "less than", "outperforms", "compared with", "versus")


def audit_answer_unsupported_comparisons(answer: str, evidence: Iterable[Any]) -> list[dict[str, Any]]:
    evidence_text = [_norm(_evidence_text(item)) for item in evidence or []]
    rows = []
    seen = set()
    for sentence in _sentences(answer):
        cue = _cue(sentence)
        key = _norm(sentence)
        if cue is None or key in seen:
            continue
        seen.add(key)
        matches = sum(1 for text in evidence_text if key in text or cue in text)
        rows.append({"comparison_sentence": sentence, "cue": cue, "evidence_match_count": matches, "severity": None if matches else "medium"})
    return rows


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", string(text) or "") if part.strip()]


def _cue(sentence: str) -> str | None:
    lowered = sentence.casefold()
    for cue in _CUES:
        if cue in lowered:
            return cue
    return None


def _evidence_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return content_text(item)


def _norm(text: Any) -> str:
    return re.sub(r"\W+", " ", string(text) or "").strip().casefold()
