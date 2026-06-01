"""Audit conditional recommendation claims in RAG answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_CONDITION_RE = re.compile(r"(?=\b(if|when|unless)\b\s+([^,.;]+))", re.I)
_ACTION_RE = re.compile(r"\b(should|must|recommend(?:ed)?|use|choose|avoid|consider)\b\s+([^.;]+)", re.I)


def audit_answer_conditional_claims(answer: str, context_items: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    context_text = " ".join(content_text(item) for item in (context_items or [])).casefold()
    rows = []
    for sentence in _sentences(answer):
        for condition in _conditions(sentence):
            action = _action(sentence)
            if not action:
                continue
            supported = condition.casefold() in context_text and action.casefold() in context_text
            rows.append({"condition_phrase": condition, "action_phrase": action, "supported": supported, "severity": "none" if supported else "medium", "sentence": sentence})
    return rows


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(text or "")) if match.group(0).strip()]


def _conditions(sentence: str) -> list[str]:
    return [f"{match.group(1).casefold()} {match.group(2).strip()}" for match in _CONDITION_RE.finditer(sentence)]


def _action(sentence: str) -> str:
    match = _ACTION_RE.search(sentence)
    return match.group(0).strip() if match else ""
