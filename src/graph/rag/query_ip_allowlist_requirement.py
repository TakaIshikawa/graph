"""Detect IP allowlist requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_CATEGORY_SPECS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("admin_source_ip", re.compile(r"\b(?:admin(?:istration)?\s+console|admin\s+portal|management\s+console).{0,40}\b(?:source\s+ip|ip\s+restriction|trusted\s+ip)\b", re.I), "high"),
    ("allowlist", re.compile(r"\b(?:ip\s+(?:allowlist|allow\s+list|whitelist|white\s+list)|allowlist(?:ed)?\s+ips?|whitelist(?:ed)?\s+ips?)\b", re.I), "high"),
    ("blocklist", re.compile(r"\b(?:ip\s+(?:blocklist|block\s+list|blacklist|black\s+list)|blocklist(?:ed)?\s+ips?|blacklist(?:ed)?\s+ips?)\b", re.I), "medium"),
    ("cidr_range", re.compile(r"\b(?:cidr|ip\s+range|subnet\s+range|/\d{1,2}\s+range)\b", re.I), "medium"),
    ("trusted_network", re.compile(r"\b(?:trusted\s+network|corporate\s+network|office\s+network|vpn\s+only|from\s+vpn)\b", re.I), "high"),
)


def detect_query_ip_allowlist_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for category, pattern, severity in _CATEGORY_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append({"category": category, "matched_text": match.group(0), "severity": severity})
    rows.sort(key=lambda row: row["category"])
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
