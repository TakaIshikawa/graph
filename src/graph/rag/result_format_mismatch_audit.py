"""Audit whether a RAG answer matches formats requested by the query."""

from __future__ import annotations

import json
import re
from typing import Any

_FORMAT_ORDER = ("table", "json", "csv", "bullets", "checklist", "timeline", "concise_summary")
_REQUEST_PATTERNS = {
    "table": re.compile(r"\btable\b", re.IGNORECASE),
    "json": re.compile(r"\bjson\b", re.IGNORECASE),
    "csv": re.compile(r"\bcsv\b|comma-separated", re.IGNORECASE),
    "bullets": re.compile(r"\bbullets?\b|bullet points", re.IGNORECASE),
    "checklist": re.compile(r"\bchecklist\b|check list|checkbox", re.IGNORECASE),
    "timeline": re.compile(r"\btimeline\b|chronolog", re.IGNORECASE),
    "concise_summary": re.compile(r"\bconcise\b|\bbrief\b|short summary|concise summary", re.IGNORECASE),
}


def audit_result_format_mismatch(query: str, answer: str) -> dict[str, Any]:
    requested = [label for label in _FORMAT_ORDER if _REQUEST_PATTERNS[label].search(str(query or ""))]
    satisfied = [label for label in requested if _satisfies(label, str(answer or ""))]
    missing = [label for label in requested if label not in satisfied]
    warnings = [f"missing_{label}" for label in missing]
    return {
        "requested_formats": requested,
        "satisfied_formats": satisfied,
        "missing_formats": missing,
        "mismatch_count": len(missing),
        "warnings": warnings,
    }


def _satisfies(label: str, answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if label == "table":
        return any("|" in lines[index] and index + 1 < len(lines) and _is_table_delimiter(lines[index + 1]) for index in range(len(lines) - 1))
    if label == "json":
        try:
            json.loads(answer)
        except ValueError:
            return False
        return True
    if label == "csv":
        return any("," in line and len([cell for cell in line.split(",") if cell.strip()]) >= 2 for line in lines)
    if label == "bullets":
        return any(re.match(r"^[-*]\s+\S", line) for line in lines)
    if label == "checklist":
        return any(re.match(r"^[-*]\s+\[[ xX]\]\s+\S", line) for line in lines)
    if label == "timeline":
        return any(re.search(r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b", line) for line in lines)
    if label == "concise_summary":
        words = re.findall(r"\w+", answer)
        return bool(words) and len(words) <= 80 and len(lines) <= 5
    return False


def _is_table_delimiter(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(cell and "-" in cell and set(cell) <= {"-", ":"} for cell in cells)
