"""Audit chronological sequence of dated answer claims."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import iter_strings, metadata, result_date

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_ISO_RE = re.compile(r"\b((?:18|19|20)\d{2})-\d{2}-\d{2}\b")
_YEAR_RE = re.compile(r"\b((?:18|19|20)\d{2})\b")


def audit_answer_claim_sequence(answer: str, evidence: Iterable[Any] = ()) -> dict[str, Any]:
    """Check whether dated answer claims follow chronological order."""
    ordered = []
    for index, sentence in enumerate(_sentences(answer)):
        year = _claim_year(sentence)
        if year is not None:
            ordered.append({"sentence": sentence, "order": index + 1, "year": year})

    out = []
    previous = None
    for row in ordered:
        if previous is not None and row["year"] < previous:
            out.append(row)
        previous = row["year"]

    evidence_years = sorted({year for row in evidence for year in _evidence_years(row)})
    score = 1.0 if len(ordered) < 2 else round((len(ordered) - len(out)) / len(ordered), 3)
    hint = "no dated claims detected" if not ordered else "claims are chronological" if not out else "reorder dated claims chronologically"
    if evidence_years and ordered:
        hint += f"; evidence years span {evidence_years[0]}-{evidence_years[-1]}"
    return {
        "ordered_claims": ordered,
        "out_of_order_claims": out,
        "sequence_score": score,
        "chronology_hint": hint,
    }


def _sentences(answer: str) -> list[str]:
    return [" ".join(match.group(0).split()) for match in _SENTENCE_RE.finditer(answer or "") if match.group(0).strip()]


def _claim_year(sentence: str) -> int | None:
    match = _ISO_RE.search(sentence) or _YEAR_RE.search(sentence)
    return int(match.group(1)) if match else None


def _evidence_years(row: Any) -> list[int]:
    parsed = result_date(row)
    years = [parsed.year] if parsed else []
    for text in iter_strings(metadata(row)):
        years.extend(int(match.group(1)) for match in _YEAR_RE.finditer(text))
    return years
