"""Extract methodology-related signals from RAG results."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id

_SIGNALS = {
    "sample_size": re.compile(r"\b(?:n\s*=\s*\d+|sample size|participants?|respondents?|subjects?)\b", re.I),
    "dataset": re.compile(r"\b(dataset|data set|corpus|records?|observations?)\b", re.I),
    "measurement": re.compile(r"\b(measured|metric|measurement|confidence interval|p-value|accuracy|precision|recall)\b", re.I),
    "benchmark": re.compile(r"\b(benchmark|test set|evaluation|experiment|trial|baseline)\b", re.I),
    "survey": re.compile(r"\b(survey|interview|questionnaire|focus group)\b", re.I),
}


def extract_result_methodology_signals(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Return methodology signal rows for result-like objects."""
    rows = []
    for index, result in enumerate(results):
        text = " ".join([content_text(result), " ".join(iter_strings(metadata(result)))])
        signals = [name for name, pattern in _SIGNALS.items() if pattern.search(text)]
        warnings = [] if signals else ["missing_methodology_signals"]
        score = round(min(1.0, len(signals) / 4), 3)
        rows.append(
            {
                "result_id": result_id(result, index),
                "signals": signals,
                "warnings": warnings,
                "methodology_score": score,
            }
        )
    return rows
