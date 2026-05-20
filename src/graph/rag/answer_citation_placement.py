"""Audit citation placement near factual claim cues in RAG answers."""

from __future__ import annotations

import re
from typing import Any

_DEFAULT_CITATION_PATTERNS = (
    r"\[\d+\]",
    r"\[[^\]]+\]\([^)]+\)",
    r"\([A-Za-z][^)]*,\s*\d{4}\)",
)
_NUMBER_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:[.,]\d+)?%?|\d+(?:[.,]\d+)?%)(?!\d)")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    re.I,
)
_ABSOLUTE_RE = re.compile(r"\b(always|never|all|none|guaranteed|proves|must|only)\b", re.I)
_HIGH_STAKES_RE = re.compile(r"\b(medical|diagnosis|dose|legal|lawsuit|contract|financial|investment|tax|loan)\b", re.I)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def audit_answer_citation_placement(answer: str, *, citation_patterns: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    """Return sentence-level citation status for factual claim cues."""
    text = str(answer or "").strip()
    if not text:
        return {
            "sentence_count": 0,
            "claim_sentence_count": 0,
            "cited_claim_count": 0,
            "uncited_claim_count": 0,
            "claim_type_counts": _empty_counts(),
            "sentences": [],
            "warnings": ["no_answer"],
        }

    citation_re = _citation_regex(citation_patterns)
    records = []
    for index, sentence in enumerate(_sentences(text)):
        claim_text = citation_re.sub("", sentence)
        cues = _claim_cues(claim_text)
        if not cues:
            continue
        has_citation = bool(citation_re.search(sentence))
        status = "cited" if has_citation else "uncited"
        records.append(
            {
                "sentence_index": index,
                "citation_status": status,
                "claim_types": cues,
                "evidence": sentence[:160],
            }
        )

    counts = {key: sum(1 for row in records if key in row["claim_types"]) for key in _empty_counts()}
    warnings = []
    uncited_types = {claim_type for row in records if row["citation_status"] == "uncited" for claim_type in row["claim_types"]}
    for claim_type in ("numeric_claim", "date_claim", "absolute_claim", "high_stakes_domain"):
        if claim_type in uncited_types:
            warnings.append(f"uncited_{claim_type}")

    return {
        "sentence_count": len(_sentences(text)),
        "claim_sentence_count": len(records),
        "cited_claim_count": sum(1 for row in records if row["citation_status"] == "cited"),
        "uncited_claim_count": sum(1 for row in records if row["citation_status"] == "uncited"),
        "claim_type_counts": counts,
        "sentences": records,
        "warnings": warnings,
    }


def _citation_regex(patterns: list[str] | tuple[str, ...] | None) -> re.Pattern[str]:
    selected = _DEFAULT_CITATION_PATTERNS if patterns is None else tuple(patterns)
    if not selected:
        return re.compile(r"a\A")
    return re.compile("|".join(f"(?:{pattern})" for pattern in selected))


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _claim_cues(text: str) -> list[str]:
    cues = []
    for label, pattern in (
        ("numeric_claim", _NUMBER_RE),
        ("date_claim", _DATE_RE),
        ("absolute_claim", _ABSOLUTE_RE),
        ("high_stakes_domain", _HIGH_STAKES_RE),
    ):
        if pattern.search(text):
            cues.append(label)
    return cues


def _empty_counts() -> dict[str, int]:
    return {"numeric_claim": 0, "date_claim": 0, "absolute_claim": 0, "high_stakes_domain": 0}
