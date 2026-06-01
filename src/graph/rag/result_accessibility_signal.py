"""Analyze accessibility and access signals in result records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, value

_FIELDS = ("paywall", "login_required", "open_access", "pdf_url", "transcript_url", "alt_text", "captions", "accessibility")
_ACCESSIBLE = ("open_access", "pdf_url", "transcript_url", "alt_text", "captions")
_RESTRICTED = ("paywall", "login_required")


def analyze_result_accessibility_signals(results: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    rows = list(results or [])
    counts: Counter[str] = Counter()
    accessible = restricted = unknown = 0
    samples = []
    for index, result in enumerate(rows):
        signals = _signals(result)
        counts.update(signals)
        if any(signal in signals for signal in _RESTRICTED):
            restricted += 1
            status = "restricted"
        elif any(signal in signals for signal in _ACCESSIBLE):
            accessible += 1
            status = "accessible"
        else:
            unknown += 1
            status = "unknown"
        if len(samples) < sample_limit:
            samples.append({"result_id": result_id(result, index), "title": string(value(result, "title")) or "", "status": status, "signals": signals})
    return {"accessible_count": accessible, "restricted_count": restricted, "unknown_count": unknown, "signal_counts": {field: counts.get(field, 0) for field in _FIELDS}, "samples": samples}


def _signals(result: Any) -> list[str]:
    found = []
    for key in _FIELDS:
        raw = value(result, key)
        if raw is True or (isinstance(raw, str) and raw.strip()):
            found.append(key)
    return found
