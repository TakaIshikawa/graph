"""Score conflict severity across evidence snippets."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from graph.rag._analysis_utils import content_text, ordered_terms, result_id, source_id

_POSITIVE_RE = re.compile(r"\b(?:increase[sd]?|higher|improved?|effective|supports?|allowed|available|safe)\b", re.I)
_NEGATIVE_RE = re.compile(r"\b(?:decrease[sd]?|lower|worse|ineffective|unsupported|blocked|unavailable|unsafe|not supported)\b", re.I)
_NUMBER_RE = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?%?(?![\w.-])")
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b")


def score_evidence_conflict_severity(results: Iterable[Any]) -> dict[str, Any]:
    """Return conflicts, severity counts, and max severity across result pairs."""
    rows = [_analyze(result, index) for index, result in enumerate(results)]
    conflicts = []
    for left, right in combinations(rows, 2):
        shared = sorted(set(left["terms"]) & set(right["terms"]))
        if not shared:
            continue
        reasons = []
        severity = "low"
        if left["polarity"] and right["polarity"] and left["polarity"] != right["polarity"]:
            reasons.append("opposing_polarity")
            severity = "medium"
        if left["numbers"] and right["numbers"] and set(left["numbers"]) != set(right["numbers"]):
            reasons.append("numeric_mismatch")
            severity = "high"
        if left["dates"] and right["dates"] and set(left["dates"]) != set(right["dates"]):
            reasons.append("date_mismatch")
            severity = "high" if severity == "high" else "medium"
        if not reasons:
            continue
        conflicts.append(
            {
                "topic_terms": shared[:5],
                "result_ids": [left["result_id"], right["result_id"]],
                "source_ids": [left["source_id"], right["source_id"]],
                "severity": severity,
                "reasons": reasons,
            }
        )
    counts = Counter(conflict["severity"] for conflict in conflicts)
    max_severity = _max_severity(counts)
    return {"conflicts": conflicts, "severity_counts": dict(counts), "max_severity": max_severity}


def classify_evidence_conflict_severity(conflicts: list[dict] | list[str]) -> dict[str, Any]:
    """Classify explicit conflict descriptions as low, medium, or high severity."""
    classifications = []
    counts: Counter[str] = Counter()
    for index, conflict in enumerate(conflicts):
        text = content_text(conflict) if isinstance(conflict, dict) else str(conflict)
        severity, reasons = _classify_text(text)
        counts[severity] += 1
        classifications.append({"index": index, "severity": severity, "reasons": reasons, "text": text})
    return {"severity_counts": dict(counts), "classifications": classifications}


def _classify_text(text: str) -> tuple[str, list[str]]:
    reasons = []
    lowered = text.casefold()
    severity = "low"
    if re.search(r"\b(direct contradiction|contradicts?|opposite|cannot both be true)\b", lowered):
        reasons.append("direct_contradiction")
        severity = "high"
    if re.search(r"\b(numeric disagreement|number mismatch|large difference|[0-9].*(?:vs|versus).*[0-9])\b", lowered):
        reasons.append("numeric_disagreement")
        severity = "high"
    if re.search(r"\b(date mismatch|different dates?|timing mismatch)\b", lowered):
        reasons.append("date_mismatch")
        severity = "high" if severity == "high" else "medium"
    if re.search(r"\b(source disagreement|sources disagree|publisher disagreement)\b", lowered):
        reasons.append("source_disagreement")
        severity = "high" if severity == "high" else "medium"
    if not reasons:
        reasons.append("minor_wording_difference")
    return severity, reasons


def _analyze(result: Any, index: int) -> dict[str, Any]:
    text = content_text(result)
    polarity = None
    if _POSITIVE_RE.search(text):
        polarity = "positive"
    if _NEGATIVE_RE.search(text):
        polarity = "negative" if polarity is None else "mixed"
    return {
        "result_id": result_id(result, index),
        "source_id": source_id(result),
        "terms": ordered_terms(text, min_length=4),
        "polarity": polarity,
        "numbers": _NUMBER_RE.findall(text),
        "dates": _DATE_RE.findall(text),
    }


def _max_severity(counts: Counter[str]) -> str:
    for level in ("high", "medium", "low"):
        if counts.get(level, 0):
            return level
    return "none"
