"""Audit procedural answers for step ordering issues."""

from __future__ import annotations

import re
from typing import Any

_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s+", re.M)
_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "finally": 99}
_WORD_RE = re.compile(r"\b(first|second|third|fourth|fifth|finally)\b", re.I)


def audit_answer_step_order(answer: str) -> dict[str, Any]:
    text = str(answer or "")
    numbers = [int(match.group(1)) for match in _NUMBERED_RE.finditer(text)]
    issues = []
    if numbers:
        seen = set()
        for number in numbers:
            if number in seen:
                issues.append({"type": "duplicate_step_number", "step": number})
            seen.add(number)
        for prev, cur in zip(numbers, numbers[1:], strict=False):
            if cur < prev:
                issues.append({"type": "decreasing_step_number", "from": prev, "to": cur})
            elif cur > prev + 1:
                issues.append({"type": "skipped_step_number", "from": prev, "to": cur})
    markers = [(match.group(1).casefold(), _WORDS[match.group(1).casefold()]) for match in _WORD_RE.finditer(text)]
    for (left_word, left), (right_word, right) in zip(markers, markers[1:], strict=False):
        if right < left:
            issues.append({"type": "conflicting_sequence_marker", "from": left_word, "to": right_word})
    return {"ordered": not issues, "step_numbers": numbers, "sequence_markers": [word for word, _ in markers], "issues": issues}
