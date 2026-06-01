"""Score specificity cues in evidence records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import record_id, text_blob

_FEATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("comparison", re.compile(r"\b(?:higher|lower|increase|decrease|versus|compared with|more than|less than)\b", re.I)),
    ("date", re.compile(r"\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b")),
    ("named_entity", re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")),
    ("numeric_value", re.compile(r"\b\d+(?:\.\d+)?%?\b")),
    ("quoted_span", re.compile(r'"[^"]{4,}"|\'[^\']{4,}\'')),
)


def score_evidence_claim_specificity(evidence: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    records = list(evidence or ())
    feature_counts: Counter[str] = Counter()
    samples = []
    scores = []
    for index, item in enumerate(records):
        text = text_blob(item)
        features = [name for name, pattern in _FEATURES if pattern.search(text)]
        for name in features:
            feature_counts[name] += 1
        score = len(features)
        scores.append(score)
        if len(samples) < sample_limit:
            samples.append({"result_id": record_id(item, index, "evidence"), "score": score, "features": features})
    return {
        "record_count": len(records),
        "average_specificity_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "high_specificity_count": sum(1 for score in scores if score >= 3),
        "low_specificity_count": sum(1 for score in scores if score <= 1),
        "feature_counts": {name: feature_counts.get(name, 0) for name, _ in _FEATURES},
        "samples": samples,
    }
