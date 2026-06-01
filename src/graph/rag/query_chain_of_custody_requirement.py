"""Detect chain-of-custody requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CONTROL_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("chain_of_custody", "high", (r"\bchain[-\s]of[-\s]custody\b", r"\bcustody\s+chain\b")),
    ("custody_records", "high", (r"\bcustody\s+records?\b", r"\bevidence\s+custody\s+log\b")),
    ("evidence_handling", "medium", (r"\bevidence\s+handling\b", r"\bhandle\s+evidence\b")),
    ("tamper_evident_logs", "high", (r"\btamper[-\s]evident\s+logs?\b", r"\bimmutable\s+custody\s+logs?\b")),
    ("audit_trail_preservation", "medium", (r"\baudit\s+trail\s+preservation\b", r"\bpreserve\s+the\s+audit\s+trail\b")),
)


def detect_query_chain_of_custody_requirement(query: str) -> dict[str, Any]:
    """Return chain-of-custody controls mentioned by a query."""
    text = " ".join(str(query or "").split())
    controls = [
        control for control, _severity, patterns in _CONTROL_SPECS if any(re.search(pattern, text, re.I) for pattern in patterns)
    ]
    severity = "high" if any(control in {"chain_of_custody", "custody_records", "tamper_evident_logs"} for control in controls) else "medium" if controls else "none"
    return {
        "requires_chain_of_custody": bool(controls),
        "custody_controls": controls,
        "matched_cues": controls,
        "severity": severity,
    }
