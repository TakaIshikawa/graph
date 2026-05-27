"""Detect retraction and correction signals in RAG evidence."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, value

_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("expression_of_concern", re.compile(r"\bexpression\s+of\s+concern\b", re.I)),
    ("retraction", re.compile(r"\b(?:retracted|retraction|retracts)\b", re.I)),
    ("withdrawal", re.compile(r"\b(?:withdrawn|withdrawal|withdraws)\b", re.I)),
    ("correction", re.compile(r"\b(?:corrected|correction|erratum|corrigendum)\b", re.I)),
)


def audit_evidence_retraction_signals(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return evidence items with retraction-like status cues."""
    counts = {name: 0 for name, _ in _SIGNALS}
    flagged = []
    for index, item in enumerate(results or []):
        text = _combined_text(item)
        for signal_type, pattern in _SIGNALS:
            match = pattern.search(text)
            if match:
                counts[signal_type] += 1
                flagged.append(
                    {
                        "source_id": result_id(item, index),
                        "title": string(value(item, "title")),
                        "signal_type": signal_type,
                        "cue": match.group(0),
                    }
                )
    return {"has_retraction_signals": bool(flagged), "signal_counts": counts, "flagged_results": flagged}


def _combined_text(item: Any) -> str:
    parts = [content_text(item)]
    for key in ("status", "notes"):
        text = string(value(item, key))
        if text:
            parts.append(text)
    return " ".join(parts)
