"""Audit measurement unit consistency in answers."""

from __future__ import annotations

import re
from typing import Any

_FAMILIES = {
    "currency": (("$", r"\$"), ("USD", r"\bUSD\b"), ("EUR", r"\bEUR\b"), ("GBP", r"\bGBP\b")),
    "distance": (("km", r"\bkm\b"), ("km", r"\bkilometers?\b"), ("mi", r"\bmi\b"), ("mi", r"\bmiles?\b")),
    "change": (("%", r"%"), ("percent", r"\bpercent(?:age)?\b"), ("basis_points", r"\bbasis points?\b|\bbps\b")),
}
_CONVERSION_RE = re.compile(r"\b(convert|equivalent|equals?|roughly|about)\b", re.I)


def audit_answer_unit_consistency(answer: str) -> dict[str, Any]:
    text = str(answer or "")
    issues = []
    for family, patterns in _FAMILIES.items():
        units = []
        snippets = []
        for label, pattern in patterns:
            if re.search(pattern, text, re.I):
                units.append(label)
                snippets.append(_snippet(text, pattern))
        units = list(dict.fromkeys(units))
        if len(units) > 1 and not _CONVERSION_RE.search(text):
            issues.append({"unit_family": family, "units": units, "snippets": snippets, "severity": "medium"})
    return {"issues": issues, "issue_count": len(issues)}


def _snippet(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.I)
    if not match:
        return ""
    return text[max(0, match.start() - 24) : match.end() + 24].strip()
