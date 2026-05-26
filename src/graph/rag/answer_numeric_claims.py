"""Audit numeric claims in an answer against evidence snippets."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_NUMBER_RE = re.compile(r"(?<![\w.])(\d{4}|\d+(?:\.\d+)?%?)(?!\w)")


def audit_answer_numeric_claims(answer: str, evidence: Iterable[Any]) -> list[dict[str, Any]]:
    evidence_text = "\n".join(_text(item) for item in evidence)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in _NUMBER_RE.finditer(str(answer)):
        claim_text = _sentence(str(answer), match.start(), match.end())
        normalized = _normalize(match.group(1))
        key = (claim_text, normalized)
        if key in seen:
            continue
        seen.add(key)
        count = len(re.findall(rf"(?<![\w.]){re.escape(match.group(1))}(?!\w)", evidence_text, re.IGNORECASE))
        rows.append({"claim_text": claim_text, "normalized_number": normalized, "evidence_match_count": count, "severity": "none" if count else "medium"})
    return sorted(rows, key=lambda row: (row["claim_text"], row["normalized_number"]))


def _normalize(value: str) -> str:
    return value.rstrip("%") + ("%" if value.endswith("%") else "")


def _sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    right_dot = text.find(".", end)
    right_newline = text.find("\n", end)
    candidates = [pos for pos in (right_dot, right_newline) if pos != -1]
    right = min(candidates) if candidates else len(text)
    return " ".join(text[left:right].split())


def _text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("snippet", "text", "content"):
            if item.get(key) not in (None, ""):
                return str(item[key])
    for key in ("snippet", "text", "content"):
        value = getattr(item, key, None)
        if value not in (None, ""):
            return str(value)
    return ""
