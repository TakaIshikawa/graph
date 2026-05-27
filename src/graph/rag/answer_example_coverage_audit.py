"""Audit whether example-bearing answer sentences are covered by evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, string

_CUES = ("for example", "e.g.", "such as", "including")


def audit_answer_example_coverage(answer: str, evidence: Iterable[Any]) -> list[dict[str, Any]]:
    evidence_texts = [_norm(_evidence_text(item)) for item in evidence or []]
    rows = []
    seen = set()
    for sentence in _sentences(answer):
        cue = _cue(sentence)
        key = _norm(sentence)
        if cue is None or key in seen:
            continue
        seen.add(key)
        tail = _norm(_example_tail(sentence, cue))
        needle = tail or key
        matches = sum(1 for text in evidence_texts if key in text or needle in text)
        rows.append({"example_sentence": sentence, "cue": cue, "evidence_match_count": matches, "severity": None if matches else "medium"})
    return rows


def _sentences(text: str) -> list[str]:
    protected = re.sub(r"e\.g\.", "e<dot>g<dot>", string(text) or "", flags=re.I)
    return [part.strip().replace("e<dot>g<dot>", "e.g.") for part in re.split(r"(?<=[.!?])\s+", protected) if part.strip()]


def _cue(sentence: str) -> str | None:
    lowered = sentence.casefold()
    for cue in _CUES:
        if cue in lowered:
            return cue
    return None


def _example_tail(sentence: str, cue: str) -> str:
    return sentence.casefold().split(cue, 1)[-1]


def _evidence_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return content_text(item)


def _norm(text: Any) -> str:
    return re.sub(r"\W+", " ", string(text) or "").strip().casefold()
