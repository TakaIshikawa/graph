"""Analyze evidence records for access barriers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id, string, value

_BARRIERS: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    ("paywall", "high", "Prefer open-access copies, preprints, or summaries from the same source.", re.compile(r"\b(?:paywall|subscribe|subscription|purchase|rent|buy article|paid access)\b", re.I)),
    ("login_required", "medium", "Look for a public mirror or cite only metadata that is visible without login.", re.compile(r"\b(?:login required|sign in|account required|registration required)\b", re.I)),
    ("abstract_only", "medium", "Treat claims as limited unless full text or methods are available.", re.compile(r"\b(?:abstract only|abstract available|no full text)\b", re.I)),
    ("blocked_pdf", "medium", "Retrieve an HTML version or alternate PDF host before relying on page details.", re.compile(r"\b(?:pdf blocked|cannot access pdf|forbidden pdf|403|blocked pdf)\b", re.I)),
    ("missing_full_text", "medium", "Find a full-text source before using detailed claims.", re.compile(r"\b(?:full text unavailable|missing full text|no full text|citation only)\b", re.I)),
    ("archived_copy_available", "low", "Use the archived copy to preserve access and capture date.", re.compile(r"\b(?:archived copy|wayback|internet archive|perma\.cc)\b", re.I)),
)


def analyze_evidence_access_barriers(evidence: Iterable[Any]) -> list[dict[str, Any]]:
    """Return access barrier labels, severities, and mitigation hints."""
    return [_analyze(record, index) for index, record in enumerate(evidence or [])]


def _analyze(record: Any, index: int) -> dict[str, Any]:
    text = _record_text(record)
    barriers = []
    for label, severity, hint, pattern in _BARRIERS:
        if pattern.search(text):
            barriers.append({"label": label, "severity": severity, "mitigation": hint})
    severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    overall = max((barrier["severity"] for barrier in barriers), key=severity_rank.get, default="none")
    return {
        "evidence_id": result_id(record, index),
        "barrier_labels": [barrier["label"] for barrier in barriers],
        "severity": overall,
        "mitigation_hints": [barrier["mitigation"] for barrier in barriers],
    }


def _record_text(record: Any) -> str:
    parts = [
        content_text(record),
        string(value(record, "url")) or "",
        string(value(record, "source")) or "",
        " ".join(iter_strings(metadata(record))),
    ]
    return " ".join(parts)
