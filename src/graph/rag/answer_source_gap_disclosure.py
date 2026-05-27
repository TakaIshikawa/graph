"""Audit whether answers disclose missing source categories."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_GAP_RE = re.compile(
    r"\b(?:no|not|without|missing|absent|unavailable|exclude[sd]?|omitted|lack(?:ing)?|did\s+not\s+include|could\s+not\s+access)\b",
    re.I,
)
_SOURCE_RE = re.compile(r"\b(?:source|sources|data|dataset|datasets|records|logs|documents|docs|evidence|archive|archives|citations)\b", re.I)


def audit_answer_source_gap_disclosure(answer: str, missing_sources: Iterable[str] | None = None) -> dict[str, Any]:
    """Return source-gap disclosure signals and undisclosed missing sources."""
    text = _inline_text(answer)
    disclosures = _disclosures(text)
    missing = [_inline_text(source) for source in missing_sources or () if _inline_text(source)]
    disclosed_missing = [source for source in missing if _mentions_source_gap(text, source)]
    undisclosed = [source for source in missing if source not in disclosed_missing]
    findings = [
        {
            "source": source,
            "issue": "missing_source_not_disclosed",
            "message": f"Disclose that {source} sources are missing, unavailable, or excluded.",
        }
        for source in undisclosed
    ]
    return {
        "disclosure_count": len(disclosures),
        "disclosures": disclosures,
        "missing_sources": missing,
        "disclosed_missing_sources": disclosed_missing,
        "undisclosed_missing_sources": undisclosed,
        "findings": findings,
        "coverage_score": 1.0 if not missing else round(len(disclosed_missing) / len(missing), 3),
    }


def _disclosures(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sentence in _sentences(text):
        if _GAP_RE.search(sentence) and _SOURCE_RE.search(sentence):
            rows.append({"text": sentence, "gap_terms": _matches(_GAP_RE, sentence)})
    return rows


def _mentions_source_gap(text: str, source: str) -> bool:
    source_pattern = re.compile(rf"\b{re.escape(source)}\b", re.I)
    for sentence in _sentences(text):
        if source_pattern.search(sentence) and _GAP_RE.search(sentence):
            return True
    return False


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return [match.group(0) for match in pattern.finditer(text)]


def _sentences(text: str) -> list[str]:
    return [part.strip(" -") for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip(" -")]


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
