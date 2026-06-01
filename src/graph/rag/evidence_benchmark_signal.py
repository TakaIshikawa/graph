"""Analyze evidence for benchmark and evaluation signals."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._record_text import record_id, text_blob

_BENCHMARK = re.compile(r"\b(?:benchmark|baseline|leaderboard|evaluation results?)\b", re.I)
_METRICS: tuple[tuple[str, str], ...] = (("accuracy", r"\baccuracy\b"), ("f1", r"\bf1\b"), ("latency", r"\blatency\b"), ("score", r"\bscore\b"))
_DATASETS: tuple[tuple[str, str], ...] = (("dataset", r"\bdatasets?\b"), ("test_set", r"\btest\s+set\b"), ("validation_set", r"\bvalidation\s+set\b"))


def analyze_evidence_benchmark_signals(evidence: Iterable[Any]) -> dict[str, Any]:
    count = 0
    metrics: set[str] = set()
    datasets: set[str] = set()
    samples = []
    for index, item in enumerate(evidence):
        text = text_blob(item)
        item_metrics = [name for name, pattern in _METRICS if re.search(pattern, text, re.I)]
        item_datasets = [name for name, pattern in _DATASETS if re.search(pattern, text, re.I)]
        if _BENCHMARK.search(text) or item_metrics or item_datasets:
            count += 1
            metrics.update(item_metrics)
            datasets.update(item_datasets)
            samples.append({"source_id": record_id(item, index, "evidence"), "signals": sorted(set(item_metrics + item_datasets))})
    return {
        "benchmark_evidence_count": count,
        "metric_terms": sorted(metrics),
        "dataset_terms": sorted(datasets),
        "benchmark_samples": samples[:5],
    }
