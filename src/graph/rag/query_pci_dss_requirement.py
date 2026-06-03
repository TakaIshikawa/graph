"""Detect PCI DSS requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pci_dss", "high", (r"\bpci\s*dss\b", r"\bpayment\s+card\s+industry\s+data\s+security\s+standard\b")),
    ("cardholder_data_environment", "high", (r"\bcardholder\s+data\s+environment\b", r"\bcde\b")),
    ("pan_card_storage", "high", (r"\bprimary\s+account\s+number\b", r"\bpan\s+(?:handling|storage|data)\b", r"\bstore\s+(?:credit\s+)?card\s+(?:numbers?|data)\b")),
    ("saq", "medium", (r"\bsaq(?:-[a-z0-9]+)?\b", r"\bself-assessment\s+questionnaire\b")),
    ("merchant_level", "medium", (r"\bmerchant\s+levels?\s+[1-4]\b", r"\blevel\s+[1-4]\s+merchant\b")),
    ("tokenization_scope", "medium", (r"\btokeni[sz]ed?\s+card\s+storage\b", r"\bpayment\s+processor\s+scope\b", r"\bcard\s+tokeni[sz]ation\b")),
)


def detect_query_pci_dss_requirement(query: str) -> dict[str, Any]:
    matches = _matches(query)
    categories = sorted(dict.fromkeys(match["category"] for match in matches))
    return {"requires_pci_dss": bool(matches), "categories": categories, "matches": matches}


def _matches(query: str) -> list[dict[str, Any]]:
    text = " ".join(str(query or "").split())
    rows = []
    for category, severity, patterns in _PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                rows.append({"matched_text": match.group(0), "category": category, "severity": severity, "span": match.span()})
    return sorted(rows, key=lambda row: (row["span"][0], row["category"], row["matched_text"].casefold()))
