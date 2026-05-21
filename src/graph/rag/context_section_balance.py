"""Analyze section balance across retrieved RAG context results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, rounded_ratio, string, value

_SECTION_KEYS = ("section", "heading", "category", "source_type", "collection")


def analyze_context_section_balance(
    results: Iterable[Any],
    *,
    section_key: str = "section",
    dominance_threshold: float = 0.6,
) -> dict[str, Any]:
    """Count normalized section labels and flag over-concentrated context."""
    rows = list(results or [])
    result_rows = []
    counts: Counter[str] = Counter()
    for index, result in enumerate(rows):
        section, reason = _section_label(result, section_key)
        counts[section] += 1
        result_rows.append({"result_id": result_id(result, index), "section": section, "reasons": [] if reason is None else [reason]})

    total = len(rows)
    dominant_section = None
    dominant_share = 0.0
    if counts:
        dominant_section, dominant_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        dominant_share = rounded_ratio(dominant_count, total)

    if dominant_section and dominant_share > dominance_threshold:
        for row in result_rows:
            if row["section"] == dominant_section:
                row["reasons"].append("dominant_section_overrepresented")

    underrepresented = [
        section for section, count in sorted(counts.items()) if total and rounded_ratio(count, total) < 0.2
    ]
    reason_counts = _reason_counts(result_rows)
    warnings = ["no_results"] if not rows else []
    if "dominant_section_overrepresented" in reason_counts:
        warnings.append("dominant_section_overrepresented")

    return {
        "total_results": total,
        "section_counts": dict(sorted(counts.items())),
        "dominant_section": dominant_section,
        "dominant_share": dominant_share,
        "underrepresented_sections": underrepresented,
        "results": result_rows,
        "reason_counts": reason_counts,
        "warnings": warnings,
    }


def _section_label(result: Any, preferred_key: str) -> tuple[str, str | None]:
    for key in (preferred_key, *_SECTION_KEYS):
        text = string(value(result, key))
        if text:
            return "_".join(text.casefold().split()), None
    return "unknown", "missing_section_metadata"


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter(reason for row in rows for reason in row["reasons"])
    return dict(sorted(counter.items()))
