"""Detect integration surface requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SURFACE_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("api", "high", re.compile(r"\b(?:rest\s+api|graphql\s+api|http\s+api|api|endpoint)\b", re.I)),
    ("webhook", "high", re.compile(r"\bwebhooks?\b", re.I)),
    ("cli", "medium", re.compile(r"\b(?:cli|command[-\s]line|terminal\s+command)\b", re.I)),
    ("sdk", "medium", re.compile(r"\b(?:sdk|client\s+library|software\s+development\s+kit)\b", re.I)),
    ("mcp_server", "high", re.compile(r"\b(?:mcp\s+server|model\s+context\s+protocol\s+server)\b", re.I)),
    ("database_connector", "high", re.compile(r"\b(?:database\s+connector|db\s+connector|sql\s+connector|jdbc|odbc)\b", re.I)),
    ("file_import", "medium", re.compile(r"\b(?:file\s+import|import\s+(?:files?|csv|json|spreadsheet|uploads?)|bulk\s+import)\b", re.I)),
    ("file_export", "medium", re.compile(r"\b(?:file\s+export|export\s+(?:files?|csv|json|spreadsheet|data)|bulk\s+export)\b", re.I)),
)


def detect_query_integration_surface_requirements(query: str) -> list[dict[str, Any]]:
    normalized = _normalize_query(query)
    rows = []
    for surface, severity, pattern in _SURFACE_SPECS:
        match = pattern.search(normalized)
        if match:
            rows.append(
                {
                    "matched_text": match.group(0),
                    "surface": surface,
                    "severity": severity,
                    "span": [match.start(), match.end()],
                }
            )
    rows.sort(key=lambda row: (row["span"][0], row["span"][1], row["surface"]))
    return rows


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
