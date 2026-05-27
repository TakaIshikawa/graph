"""Audit stable identifier coverage for RAG evidence items."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, rounded_ratio, string, value

_IDENTIFIER_KEYS = ("id", "source_id", "url", "doi", "citation_key")


def audit_evidence_identifier_coverage(evidence: Iterable[Any]) -> dict[str, Any]:
    """Return identifier coverage counts and deterministic samples."""
    items = list(evidence or [])
    counts = {key: 0 for key in _IDENTIFIER_KEYS}
    identified = 0
    samples = []
    for index, item in enumerate(items):
        present = []
        for key in _IDENTIFIER_KEYS:
            if string(value(item, key)):
                counts[key] += 1
                present.append(key)
        if present:
            identified += 1
        elif len(samples) < 3:
            samples.append({"index": index, "snippet": (content_text(item) or string(item) or "")[:80]})

    unidentified = len(items) - identified
    return {
        "evidence_count": len(items),
        "identified_count": identified,
        "unidentified_count": unidentified,
        "coverage_ratio": rounded_ratio(identified, len(items)),
        "counts_by_identifier_type": counts,
        "samples": samples,
    }
