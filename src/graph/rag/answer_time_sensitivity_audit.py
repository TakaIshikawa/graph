"""Audit current-state answer claims for visible evidence date support."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, parse_date

_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_CURRENT_RE = re.compile(r"\b(?:currently|now|latest|today|remains|still|no longer|as of)\b", re.I)
_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_RE = re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b", re.I)


def audit_answer_time_sensitivity(answer: str, evidence: Iterable[Any]) -> dict[str, Any]:
    """Flag current-state claims lacking detectable date support in evidence."""
    answer_text = " ".join(str(answer or "").split())
    evidence_dates = sorted(_evidence_dates(evidence))
    findings = []
    answer_dates = _dates_in_text(answer_text)
    for sentence in _sentences(answer_text):
        if not _CURRENT_RE.search(sentence):
            continue
        if evidence_dates and (not answer_dates or any(item in evidence_dates for item in answer_dates)):
            continue
        findings.append(
            {
                "claim_text": sentence,
                "severity": "medium",
                "reason_codes": ["time_sensitive_claim_without_visible_date_support"],
            }
        )
    return {
        "findings": findings,
        "evidence_date_count": len(evidence_dates),
        "oldest_detected_date": evidence_dates[0].isoformat() if evidence_dates else None,
        "newest_detected_date": evidence_dates[-1].isoformat() if evidence_dates else None,
    }


def _evidence_dates(evidence: Iterable[Any]) -> set[date]:
    dates: set[date] = set()
    for record in evidence or []:
        for text in [content_text(record), *iter_strings(metadata(record))]:
            dates.update(_dates_in_text(text))
    return dates


def _dates_in_text(text: str) -> set[date]:
    found: set[date] = set()
    for match in _ISO_RE.finditer(text):
        parsed = parse_date(match.group(0))
        if parsed:
            found.add(parsed)
    for match in _MONTH_RE.finditer(text):
        parsed = _parse_month_date(match.group(0))
        if parsed:
            found.add(parsed)
    return found


def _parse_month_date(text: str) -> date | None:
    from datetime import datetime

    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]
