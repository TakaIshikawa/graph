"""Analyze credential and affiliation signals in result metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, value

_FIELDS = ("author", "byline", "organization", "affiliation", "credentials")
_SIGNALS = ("phd", "md", "professor", "university", "government", "standards body", "journal", "company")


def analyze_result_credential_signals(results: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    rows = list(results or [])
    signal_counts: Counter[str] = Counter()
    organization_counts: Counter[str] = Counter()
    with_credentials = without_credentials = 0
    samples = []
    for index, result in enumerate(rows):
        text = _credential_text(result)
        signals = [signal for signal in _SIGNALS if signal in text.casefold()]
        signal_counts.update(signals)
        org = string(value(result, "organization")) or string(value(result, "affiliation"))
        if org:
            organization_counts[org] += 1
        with_credentials += bool(signals)
        without_credentials += not bool(signals)
        if signals and len(samples) < sample_limit:
            samples.append({"result_id": result_id(result, index), "title": string(value(result, "title")) or "", "signals": signals})
    return {
        "credential_signal_counts": {signal: signal_counts.get(signal, 0) for signal in _SIGNALS},
        "results_with_credentials": with_credentials,
        "results_without_credentials": without_credentials,
        "organization_counts": dict(sorted(organization_counts.items())),
        "samples": samples,
    }


def _credential_text(result: Any) -> str:
    return " ".join(text for key in _FIELDS if (text := string(value(result, key))))
