"""Detect funding and conflict-of-interest disclosures in evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id

_CONFLICT_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("funded_by", re.compile(r"\bfunded by\b|\bsupported by\b", re.I)),
    ("industry_funded", re.compile(r"\bindustry[- ]funded\b", re.I)),
    ("sponsored_by", re.compile(r"\bsponsored by\b", re.I)),
    ("grant", re.compile(r"\bgrant (?:from|by|number)\b", re.I)),
    ("conflict_of_interest", re.compile(r"\bconflicts? of interest\b|\bcompeting interests?\b", re.I)),
    ("employment_conflict", re.compile(r"\bemployee of\b", re.I)),
    ("consulting_conflict", re.compile(r"\bconsultant for\b", re.I)),
)
_DISCLOSURE_RE = re.compile(r"\bno (?:conflicts?|competing interests?) (?:declared|reported)|nothing to disclose\b", re.I)
_FUNDER_RE = re.compile(r"\b(?:funded by|sponsored by|grant from|supported by)\s+([^.;,\n]{2,80})", re.I)


def detect_evidence_funding_conflicts(evidence: Iterable[Any]) -> dict[str, Any]:
    """Return funding conflict signals grouped by evidence id."""
    conflict_rows: list[dict[str, Any]] = []
    disclosure_rows: list[dict[str, Any]] = []
    funders: list[dict[str, str]] = []
    warnings: list[str] = []

    for index, record in enumerate(evidence or []):
        evidence_id = result_id(record, index)
        text = _record_text(record)
        cues = [label for label, pattern in _CONFLICT_CUES if pattern.search(text)]
        snippets = _funder_snippets(text)
        if cues or snippets:
            conflict_rows.append({"evidence_id": evidence_id, "conflict_cues": cues})
            if "funding_or_conflict_disclosure_present" not in warnings:
                warnings.append("funding_or_conflict_disclosure_present")
        if _DISCLOSURE_RE.search(text):
            disclosure_rows.append({"evidence_id": evidence_id, "disclosure_type": "no_conflicts_declared"})
        for snippet in snippets:
            funders.append({"evidence_id": evidence_id, "mention": snippet})

    return {
        "conflict_evidence": conflict_rows,
        "disclosure_evidence": disclosure_rows,
        "funder_mentions": funders,
        "warnings": warnings,
        "confidence": _confidence(conflict_rows, disclosure_rows),
    }


def _record_text(record: Any) -> str:
    return " ".join([content_text(record), " ".join(iter_strings(metadata(record)))])


def _funder_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()
    for match in _FUNDER_RE.finditer(text):
        snippet = " ".join(match.group(0).strip().split())[:120]
        key = snippet.casefold()
        if key not in seen:
            seen.add(key)
            snippets.append(snippet)
    return snippets


def _confidence(conflicts: list[dict[str, Any]], disclosures: list[dict[str, Any]]) -> float:
    if conflicts:
        return 0.85
    if disclosures:
        return 0.45
    return 0.0
