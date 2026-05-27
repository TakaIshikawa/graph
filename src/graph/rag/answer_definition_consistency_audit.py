"""Audit answer definitions against evidence text."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, string

_CUES = ("is defined as", "means", "refers to", "is a")
_CONFLICT = ("not", "differs", "contradicts", "conflicts")


def audit_answer_definition_consistency(answer: str, evidence: Iterable[Any]) -> list[dict[str, Any]]:
    evidence_texts = [_norm(_evidence_text(item)) for item in evidence or []]
    rows = []
    seen = set()
    for sentence in _sentences(answer):
        parsed = _definition(sentence)
        if parsed is None:
            continue
        term, definition = parsed
        key = (term, _norm(sentence))
        if key in seen:
            continue
        seen.add(key)
        term_norm = _norm(term)
        def_norm = _norm(definition)
        matches = sum(1 for text in evidence_texts if term_norm in text and (def_norm in text or _norm(sentence) in text))
        conflicts = sum(1 for text in evidence_texts if term_norm in text and any(cue in text.split() for cue in _CONFLICT))
        severity = "conflict" if conflicts else None if matches else "unsupported"
        rows.append({"term": term, "definition_sentence": sentence, "evidence_match_count": matches, "conflicting_evidence_count": conflicts, "severity": severity})
    return rows


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", string(text) or "") if part.strip()]


def _definition(sentence: str) -> tuple[str, str] | None:
    lowered = sentence.casefold()
    for cue in _CUES:
        if cue in lowered:
            before, after = re.split(re.escape(cue), sentence, maxsplit=1, flags=re.I)
            term = re.sub(r"^(?:the|a|an)\s+", "", before.strip(), flags=re.I).strip()
            return (_norm(term), after.strip().rstrip("."))
    return None


def _evidence_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return content_text(item)


def _norm(text: Any) -> str:
    return re.sub(r"\W+", " ", string(text) or "").strip().casefold()
