"""Detect policy exception requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("policy_exception", "medium", (r"\bpolicy\s+exceptions?\b", r"\bexception\s+to\s+policy\b")),
    ("waiver", "medium", (r"\bwaivers?\b", r"\bpolicy\s+waivers?\b")),
    ("compensating_control", "high", (r"\bcompensating\s+controls?\b",)),
    ("exception_approval", "medium", (r"\bexception\s+approval\b", r"\bapprove\s+the\s+exception\b")),
    ("risk_acceptance", "high", (r"\brisk\s+acceptance\b", r"\baccept(?:ed|ing)?\s+risk\b")),
    ("temporary_exemption", "medium", (r"\btemporary\s+exemptions?\b", r"\btemporary\s+exceptions?\b")),
)


def detect_query_policy_exception_requirement(query: str) -> dict[str, Any]:
    """Return policy exception requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_policy_exception": bool(matches),
        "categories": [match["category"] for match in matches],
        "matches": matches,
    }


def _detect_matches(text: str) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, (category, severity, patterns) in enumerate(_CATEGORY_SPECS):
        found = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if found:
            match = min(found, key=lambda item: item.start())
            rows.append((match.start(), index, {"category": category, "severity": severity, "matched_text": match.group(0), "span": (match.start(), match.end())}))
    return [row for _start, _index, row in sorted(rows, key=lambda item: (item[0], item[1]))]
