"""Detect log integrity requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("tamper_evidence", "high", (r"\btamper[-\s]evident\s+logs?\b", r"\blog\s+tamper\s+evidence\b")),
    ("immutable_storage", "high", (r"\bimmutable\s+logs?\b", r"\bworm\s+storage\b", r"\bwrite\s+once\s+read\s+many\b")),
    ("signing", "high", (r"\blog\s+signing\b", r"\bsigned\s+logs?\b", r"\baudit\s+log\s+integrity\b")),
    ("chain_verification", "high", (r"\blog\s+chain\s+verification\b", r"\bverify\s+log\s+chains?\b")),
)


def detect_query_log_integrity_requirement(query: str) -> dict[str, Any]:
    """Return log integrity requirements mentioned by a query."""
    text = " ".join(str(query or "").split())
    matches = _detect_matches(text)
    return {
        "requires_log_integrity": bool(matches),
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
