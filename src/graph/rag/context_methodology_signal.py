"""Analyze methodology signals in retrieved context records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, value

_SIGNALS = ("sample size", "dataset", "survey", "randomized", "interview", "benchmark", "experiment", "limitation")


def analyze_context_methodology_signals(contexts: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    rows = list(contexts or [])
    counts: Counter[str] = Counter()
    with_methodology = without_methodology = 0
    samples = []
    for index, context in enumerate(rows):
        found = [signal for signal in _SIGNALS if signal in content_text(context).casefold()]
        counts.update(found)
        with_methodology += bool(found)
        without_methodology += not bool(found)
        if found and len(samples) < sample_limit:
            samples.append({"context_id": result_id(context, index), "title": string(value(context, "title")) or "", "signals": found})
    return {
        "signal_counts": {signal: counts.get(signal, 0) for signal in _SIGNALS},
        "contexts_with_methodology": with_methodology,
        "contexts_without_methodology": without_methodology,
        "samples": samples,
    }
