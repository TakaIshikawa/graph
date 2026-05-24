"""Audit metric unit consistency in RAG answers."""

from __future__ import annotations

import re
from typing import Any

_MENTION_RE = re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|milliseconds?|s|sec|seconds?|minutes?|hours?|\$|usd|dollars?|%|percent|monthly|annual|annually|yearly)\b|(?P<money>\$)\s*(?P<money_value>\d+(?:\.\d+)?)", re.I)
_GROUPS = {
    "ms": "time",
    "millisecond": "time",
    "milliseconds": "time",
    "s": "time",
    "sec": "time",
    "second": "time",
    "seconds": "time",
    "minute": "time",
    "minutes": "time",
    "hour": "time",
    "hours": "time",
    "$": "money",
    "usd": "money",
    "dollar": "money",
    "dollars": "money",
    "%": "percent",
    "percent": "percent",
    "monthly": "rate_period",
    "annual": "rate_period",
    "annually": "rate_period",
    "yearly": "rate_period",
}
_CONVERSION_RE = re.compile(r"\b(?:convert|conversion|normalize|normalise|equivalent|per\s+year|annualized|standardized)\b", re.I)


def audit_answer_metric_unit_consistency(answer: str) -> dict[str, Any]:
    """Return extracted numeric units and likely inconsistent unit groups."""
    text = " ".join(str(answer or "").split())
    mentions: list[dict[str, Any]] = []
    groups: dict[str, set[str]] = {}
    for match in _MENTION_RE.finditer(text):
        unit = (match.group("unit") or match.group("money") or "").casefold()
        value = match.group("value") or match.group("money_value")
        group = _GROUPS.get(unit, unit)
        mentions.append({"text": match.group(0).strip(), "value": value, "unit": unit, "family": group, "span": [match.start(), match.end()]})
        groups.setdefault(group, set()).add(unit)
    has_conversion = bool(_CONVERSION_RE.search(text))
    inconsistent = [
        {"family": family, "units": sorted(units)}
        for family, units in sorted(groups.items())
        if len(units) > 1 and not has_conversion
    ]
    warnings = [f"inconsistent_{row['family']}_units_without_conversion" for row in inconsistent]
    return {"unit_mentions": mentions, "inconsistent_unit_families": inconsistent, "has_conversion_language": has_conversion, "warnings": warnings}
