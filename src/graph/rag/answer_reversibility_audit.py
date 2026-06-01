"""Audit high-impact answer actions for nearby reversibility language."""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_ACTION_RE = re.compile(r"\b(delete|migrate|rotate|revoke|overwrite|upgrade|disable)\b[^.!?]*", re.I)
_REVERSIBLE_RE = re.compile(r"\b(rollback|roll back|undo|backup|back up|restore|recovery|recover|revert)\b", re.I)


def audit_answer_reversibility(answer: str) -> list[dict[str, Any]]:
    sentences = _sentences(answer)
    rows = []
    seen: set[str] = set()
    for index, sentence in enumerate(sentences):
        for match in _ACTION_RE.finditer(sentence):
            action = match.group(0).strip(" .")
            if action.casefold() in seen:
                continue
            window = " ".join(sentences[max(0, index - 1) : index + 2])
            if _REVERSIBLE_RE.search(window):
                continue
            seen.add(action.casefold())
            rows.append({"action_text": action, "missing_reversibility_signal": True, "severity": "high"})
    return sorted(rows, key=lambda row: row["action_text"].casefold())


def _sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(str(text or "")) if m.group(0).strip()]
