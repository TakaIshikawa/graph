"""Audit answer temporal claims against citation dates."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import parse_date, result_id, string, value

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DATE_RE = re.compile(r"\b(19\d{2}|20\d{2})-\d{2}-\d{2}\b")
_CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]")
_LABEL_KEYS = ("citation_label", "label", "citation", "id", "source_id", "result_id")


def audit_answer_citation_date_consistency(answer: str, citations: list[dict]) -> dict[str, Any]:
    citation_map = _citation_map(citations)
    issues = []
    for claim in _temporal_claims(answer):
        for label in claim["citation_ids"]:
            citation = citation_map.get(label)
            cited_date = _citation_date(citation) if citation is not None else None
            if cited_date is None:
                issues.append(
                    {
                        "claim_text": claim["claim_text"],
                        "citation_id": label,
                        "cited_date": None,
                        "severity": "medium",
                        "issue_type": "missing_citation_date",
                    }
                )
                continue
            if cited_date.year != claim["year"]:
                issues.append(
                    {
                        "claim_text": claim["claim_text"],
                        "citation_id": label,
                        "cited_date": cited_date.isoformat(),
                        "severity": "high",
                        "issue_type": "date_mismatch",
                    }
                )

    return {"has_date_consistency_issues": bool(issues), "issues": issues}


def _temporal_claims(answer: Any) -> list[dict[str, Any]]:
    text = string(answer) or ""
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not (_YEAR_RE.search(sentence) or _DATE_RE.search(sentence)):
            continue
        labels = _CITATION_RE.findall(sentence)
        if not labels:
            continue
        match = _DATE_RE.search(sentence) or _YEAR_RE.search(sentence)
        if match is None:
            continue
        claims.append({"claim_text": sentence.strip(), "year": int(match.group(1)), "citation_ids": labels})
    return claims


def _citation_map(citations: Iterable[Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for index, citation in enumerate(citations or []):
        mapped[result_id(citation, index)] = citation
        for key in _LABEL_KEYS:
            label = string(value(citation, key))
            if label is not None:
                mapped[label] = citation
    return mapped


def _citation_date(citation: Any) -> Any:
    if citation is None:
        return None
    for key in ("published_at", "publication_date", "source_date", "date", "updated_at", "created_at"):
        parsed = parse_date(value(citation, key))
        if parsed is not None:
            return parsed
    return None
