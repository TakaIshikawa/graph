"""Analyze conflict and disagreement signals in retrieved results."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import first, record_id, text_blob

_SIGNALS = (
    ("retracted", "high", re.compile(r"\bretracted?\b|retraction", re.I)),
    ("correction", "high", re.compile(r"\bcorrection\b|\bcorrected\b", re.I)),
    ("contradicts", "medium", re.compile(r"\bcontradicts?\b", re.I)),
    ("disputed", "medium", re.compile(r"\bdisputed\b", re.I)),
    ("conflicting evidence", "medium", re.compile(r"\bconflicting evidence\b", re.I)),
    ("controversy", "low", re.compile(r"\bcontrovers(?:y|ial)\b", re.I)),
    ("inconsistent", "low", re.compile(r"\binconsistent\b", re.I)),
)


def analyze_result_conflict_signals(results: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    result_count = results_with = 0
    signal_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        result_count += 1
        text = text_blob(result)
        matched = [(signal, severity) for signal, severity, pattern in _SIGNALS if pattern.search(text)]
        if matched:
            results_with += 1
        for signal, severity in matched:
            signal_counts[signal] += 1
            severity_counts[severity] += 1
            if len(samples) < limit:
                samples.append(
                    {
                        "result_id": record_id(result, index),
                        "title": str(first(result, ("title",)) or ""),
                        "signal": signal,
                        "severity": severity,
                        "snippet": str(first(result, ("snippet", "content", "text")) or "")[:160],
                    }
                )
    return {
        "result_count": result_count,
        "results_with_conflict_signal": results_with,
        "signal_counts": dict(sorted(signal_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "samples": samples,
    }
