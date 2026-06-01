"""Detect observability requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("alerting", "high", (r"\balerts?\b", r"\balerting\b", r"\bnotifications?\b")),
    ("audit_trail", "high", (r"\baudit\s+trails?\b", r"\baudit\s+events?\b")),
    ("dashboard", "medium", (r"\bdashboards?\b", r"\bstatus\s+page\b")),
    ("error_budget", "high", (r"\berror\s+budgets?\b",)),
    ("logging", "medium", (r"\blogs?\b", r"\blogging\b")),
    ("metrics", "medium", (r"\bmetrics?\b", r"\btelemetry\b")),
    ("slo", "high", (r"\bslos?\b", r"\bservice\s+level\s+objectives?\b")),
    ("tracing", "medium", (r"\btraces?\b", r"\btracing\b", r"\bdistributed\s+tracing\b")),
)


def detect_query_observability_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for category, severity, patterns in _CATEGORIES:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "category": category, "severity": severity})
    return sorted(rows, key=lambda row: row["category"])
