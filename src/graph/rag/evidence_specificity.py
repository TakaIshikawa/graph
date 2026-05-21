"""Score RAG results for concrete, cite-worthy evidence signals."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>'\"]+|www\.[^\s<>'\"]+")
_NUMBER_RE = re.compile(r"(?<![\w.-])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:%|x|ms|s|kg|km|m|MB|GB|USD|\$)?\b")
_DATE_RE = re.compile(
    r"(?i)\b(?:\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
_QUOTE_RE = re.compile(r'"[^"]{8,}"|\'[^\']{8,}\'|“[^”]{8,}”')
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){1,3}\b")
_METHOD_RE = re.compile(
    r"(?i)\b(?:measured|calculated from|based on|according to|using|method|sample|dataset|survey|experiment|interview|appendix|table|figure|benchmark|confidence interval)\b"
)

_WEIGHTS = {
    "number": 0.2,
    "date": 0.18,
    "url": 0.18,
    "quoted_text": 0.16,
    "named_entity": 0.16,
    "method_detail": 0.18,
}


def _bucket(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.35:
        return "medium"
    if score > 0:
        return "low"
    return "empty"


def _matches(text: str) -> dict[str, int]:
    return {
        "number": len(_NUMBER_RE.findall(text)),
        "date": len(_DATE_RE.findall(text)),
        "url": len(_URL_RE.findall(text)),
        "quoted_text": len(_QUOTE_RE.findall(text)),
        "named_entity": len(_ENTITY_RE.findall(text)),
        "method_detail": len(_METHOD_RE.findall(text)),
    }


def _score(signal_counts: Mapping[str, int]) -> float:
    total = 0.0
    for signal, weight in _WEIGHTS.items():
        count = signal_counts.get(signal, 0)
        if count:
            total += weight
            total += max(0, min(count, 3) - 1) * weight * 0.15
    return round(min(total, 1.0), 2)


def score_evidence_specificity(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return specificity scores and aggregate bucket counts for RAG results."""
    rows: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter({"empty": 0, "low": 0, "medium": 0, "high": 0})

    for index, result in enumerate(results):
        text = content_text(result)
        counts = _matches(text)
        signals = [name for name in _WEIGHTS if counts[name] > 0]
        score = _score(counts)
        bucket = _bucket(score)
        bucket_counts[bucket] += 1
        rows.append(
            {
                "result_id": result_id(result, index),
                "score": score,
                "bucket": bucket,
                "signals": signals,
                "signal_counts": {name: counts[name] for name in signals},
            }
        )

    average = round(sum(row["score"] for row in rows) / len(rows), 2) if rows else 0.0
    aggregate_bucket = _bucket(average)
    warnings: list[str] = []
    if not rows:
        warnings.append("no_results")
    elif aggregate_bucket in {"empty", "low"}:
        warnings.append("low_specificity_result_set")
    if rows and bucket_counts["high"] == 0:
        warnings.append("no_high_specificity_results")

    return {
        "result_count": len(rows),
        "average_score": average,
        "aggregate_bucket": aggregate_bucket,
        "bucket_counts": dict(bucket_counts),
        "results": rows,
        "warnings": warnings,
    }
