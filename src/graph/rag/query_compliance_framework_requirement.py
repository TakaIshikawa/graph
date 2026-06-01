"""Detect compliance-framework requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_FRAMEWORKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ccpa", "high", (r"\bccpa\b", r"\bcalifornia\s+consumer\s+privacy\s+act\b")),
    ("fedramp", "high", (r"\bfedramp\b", r"\bfed\s+ramp\b")),
    ("gdpr", "high", (r"\bgdpr\b", r"\bgeneral\s+data\s+protection\s+regulation\b")),
    ("hipaa", "high", (r"\bhipaa\b", r"\bhealth\s+insurance\s+portability\s+and\s+accountability\s+act\b")),
    ("iso_27001", "high", (r"\biso[-\s/]?27001\b", r"\biso\s*/\s*iec\s*27001\b")),
    ("pci_dss", "high", (r"\bpci[-\s]?dss\b", r"\bpayment\s+card\s+industry\s+data\s+security\s+standard\b")),
    ("soc_2", "high", (r"\bsoc\s*2\b", r"\bsoc\s*ii\b", r"\bservice\s+organization\s+control\s+2\b")),
)


def detect_query_compliance_framework_requirements(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    for framework, severity, patterns in _FRAMEWORKS:
        matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
        if matches:
            match = min(matches, key=lambda item: item.start())
            rows.append({"matched_text": match.group(0), "framework": framework, "severity": severity})
    return sorted(rows, key=lambda row: row["framework"])
