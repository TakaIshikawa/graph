"""Check unit consistency across retrieved evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_UNITS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("currency", "USD", re.compile(r"(?<!\w)(?:\$|usd\b|dollars?\b)", re.IGNORECASE)),
    ("currency", "EUR", re.compile(r"(?<!\w)(?:€|eur\b|euros?\b)", re.IGNORECASE)),
    ("currency", "GBP", re.compile(r"(?<!\w)(?:£|gbp\b|pounds?\b)", re.IGNORECASE)),
    ("percent", "percent", re.compile(r"(?<!\w)(?:%|percent(?:age)?\b)", re.IGNORECASE)),
    ("duration", "days", re.compile(r"\b\d+(?:\.\d+)?\s+days?\b", re.IGNORECASE)),
    ("duration", "hours", re.compile(r"\b\d+(?:\.\d+)?\s+hours?\b", re.IGNORECASE)),
    ("distance", "km", re.compile(r"\b\d+(?:\.\d+)?\s*(?:km|kilometers?)\b", re.IGNORECASE)),
    ("distance", "mi", re.compile(r"\b\d+(?:\.\d+)?\s*(?:mi|miles?)\b", re.IGNORECASE)),
    ("bytes", "MB", re.compile(r"\b\d+(?:\.\d+)?\s*(?:mb|megabytes?)\b", re.IGNORECASE)),
    ("bytes", "GB", re.compile(r"\b\d+(?:\.\d+)?\s*(?:gb|gigabytes?)\b", re.IGNORECASE)),
)


def check_evidence_unit_consistency(evidence: Iterable[Any]) -> dict[str, Any]:
    """Return detected measurement units and mixed-unit groups."""
    detected: list[dict[str, str]] = []
    groups: dict[str, set[str]] = {}
    for index, item in enumerate(evidence):
        text = content_text(item)
        rid = result_id(item, index)
        for group, unit, pattern in _UNITS:
            if pattern.search(text):
                detected.append({"result_id": rid, "group": group, "unit": unit})
                groups.setdefault(group, set()).add(unit)

    inconsistent = [
        {"group": group, "units": sorted(units)}
        for group, units in sorted(groups.items())
        if len(units) > 1
    ]
    hints = [f"Normalize {row['group']} values to one of: {', '.join(row['units'])}." for row in inconsistent]
    score = 1.0 if not inconsistent else max(0.0, 1.0 - len(inconsistent) * 0.35)
    return {
        "detected_units": detected,
        "inconsistent_groups": inconsistent,
        "normalization_hints": hints,
        "consistency_score": round(score, 2),
    }
