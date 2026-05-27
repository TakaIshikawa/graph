"""Extract methodology signals from evidence text."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from graph.rag._analysis_utils import content_text

_METHODS = {
    "randomized_trial": r"\brandomi[sz]ed trial\b|\brct\b",
    "survey": r"\bsurvey\b",
    "interview": r"\binterviews?\b",
    "benchmark": r"\bbenchmark\b",
    "case_study": r"\bcase stud(?:y|ies)\b",
    "systematic_review": r"\bsystematic review\b",
    "simulation": r"\bsimulation\b",
}


def extract_evidence_methodology_signals(evidence: list[dict] | list[str]) -> dict[str, Any]:
    items = []
    counts: Counter[str] = Counter()
    for index, item in enumerate(evidence):
        text = item if isinstance(item, str) else content_text(item)
        signals = [name for name, pattern in _METHODS.items() if re.search(pattern, text, re.I)]
        counts.update(signals)
        items.append({"index": index, "signals": signals})
    return {"items": items, "aggregate_counts": dict(sorted(counts.items())), "methodology_types": sorted(counts)}
