"""Analyze update-frequency and stale-maintenance signals in evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_id, string

_CADENCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("daily", (r"\bupdated\s+daily\b", r"\bdaily\s+updates?\b")),
    ("weekly", (r"\bupdated\s+weekly\b", r"\bweekly\s+updates?\b")),
    ("quarterly", (r"\bupdated\s+quarterly\b", r"\bquarterly\s+updates?\b")),
    ("annually", (r"\bupdated\s+annually\b", r"\bannual\s+updates?\b")),
    ("deprecated", (r"\bdeprecated\b",)),
    ("archived", (r"\barchived\b",)),
    ("unmaintained", (r"\bno\s+longer\s+maintained\b", r"\bunmaintained\b")),
)
_STALE = {"deprecated", "archived", "unmaintained"}


def analyze_evidence_update_frequency_signals(evidence: Iterable[Any]) -> dict[str, Any]:
    items = list(evidence or [])
    counts: Counter[str] = Counter()
    examples = []
    for index, item in enumerate(items):
        item_text = string(item) if isinstance(item, str) else content_text(item)
        text = " ".join([item_text or "", " ".join(str(v) for v in metadata(item).values())])
        for cadence, patterns in _CADENCES:
            if any(re.search(pattern, text, re.I) for pattern in patterns):
                counts[cadence] += 1
                examples.append({"source_id": result_id(item, index), "cadence": cadence})
                break
    return {
        "evidence_count": len(items),
        "cadence_counts": dict(sorted(counts.items())),
        "stale_signal_count": sum(counts[key] for key in _STALE),
        "examples": sorted(examples, key=lambda row: (row["cadence"], row["source_id"]))[:5],
    }
