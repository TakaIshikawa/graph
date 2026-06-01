"""Analyze numeric evidence signals in RAG context records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import record_id, text_blob

_PATTERNS = (
    ("p_value", re.compile(r"\bp\s*[<=>]\s*0?\.\d+\b", re.I)),
    ("sample_size", re.compile(r"\b(?:n|sample size)\s*=\s*\d+\b", re.I)),
    ("percent", re.compile(r"\b\d+(?:\.\d+)?\s*%")),
    ("currency", re.compile(r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?)|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp)\b)", re.I)),
    ("range", re.compile(r"\b\d+(?:\.\d+)?\s*(?:-|to|–)\s*\d+(?:\.\d+)?\b")),
    ("count", re.compile(r"\b\d[\d,]*\b")),
)


def analyze_context_numeric_evidence_signals(contexts: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    context_count = contexts_with = total_signals = 0
    signal_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        context_count += 1
        text = text_blob(context)
        matched_context = False
        spans: list[tuple[int, int]] = []
        for signal, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                if signal == "count" and any(match.start() >= start and match.end() <= end for start, end in spans):
                    continue
                matched_context = True
                total_signals += 1
                signal_counts[signal] += 1
                spans.append((match.start(), match.end()))
                if len(samples) < limit:
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    samples.append(
                        {
                            "context_id": record_id(context, index, "context"),
                            "signal": signal,
                            "value": match.group(0),
                            "snippet": text[start:end].strip(),
                        }
                    )
                break
        if matched_context:
            contexts_with += 1
    return {
        "context_count": context_count,
        "contexts_with_numeric_evidence": contexts_with,
        "signal_counts": dict(sorted(signal_counts.items())),
        "numeric_density": 0.0 if context_count == 0 else round(total_signals / context_count, 2),
        "samples": samples,
    }
