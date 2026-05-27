"""Audit counterargument coverage in answers."""

from __future__ import annotations

import re
from typing import Any

_EXPECT_RE = re.compile(r"\b(recommend|should|best|tradeoff|evaluate|pros and cons|whether to)\b", re.I)
_COUNTER = {
    "limitation": r"\blimitations?\b|\bhowever\b|\bbut\b",
    "downside": r"\bdownsides?\b|\brisk\b|\bdrawback\b",
    "alternative": r"\balternatives?\b|\binstead\b|\bon the other hand\b",
}


def audit_answer_counterargument_coverage(answer: str) -> dict[str, Any]:
    text = str(answer or "")
    signals = [name for name, pattern in _COUNTER.items() if re.search(pattern, text, re.I)]
    expected = bool(_EXPECT_RE.search(text)) and len(text.split()) >= 6
    return {
        "has_counterarguments": bool(signals),
        "signals": signals,
        "missing_when_expected": expected and not signals,
        "snippets": [_snippet(text, pattern) for name, pattern in _COUNTER.items() if name in signals],
    }


def _snippet(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.I)
    return "" if not match else text[max(0, match.start() - 30) : match.end() + 30].strip()
