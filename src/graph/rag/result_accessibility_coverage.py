"""Analyze accessibility evidence coverage across RAG results."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, metadata, result_id

_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("WCAG", re.compile(r"\bwcag\b|web\s+content\s+accessibility\s+guidelines", re.I)),
    ("ARIA", re.compile(r"\baria\b|aria-[a-z0-9_-]+", re.I)),
    ("alt_text", re.compile(r"\balt\s+text\b|alternative\s+text", re.I)),
    ("captions", re.compile(r"\bcaption(?:s|ed)?\b|subtitles?", re.I)),
    ("keyboard_navigation", re.compile(r"\bkeyboard\s+navigation\b|focus\s+(?:order|management)", re.I)),
    ("screen_readers", re.compile(r"\bscreen\s+readers?\b|voiceover|nvda|jaws", re.I)),
)
_ACCESS_QUERY_RE = re.compile(r"\baccessib|wcag|aria|screen\s+reader|alt\s+text|caption|keyboard\b", re.I)


def analyze_result_accessibility_coverage(results: list[dict], query: str = "") -> dict[str, Any]:
    rows = []
    for index, result in enumerate(results):
        text = f"{content_text(result)} {' '.join(str(value) for value in metadata(result).values())}"
        signals = [name for name, pattern in _SIGNALS if pattern.search(text)]
        if signals:
            rows.append({"id": result_id(result, index), "index": index, "signals": signals})
    total = len(results)
    ratio = round(len(rows) / total, 4) if total else 0.0
    recommendation = ""
    if _ACCESS_QUERY_RE.search(str(query or "")) and not rows:
        recommendation = "retrieve_sources_with_accessibility_evidence"
    return {
        "total_results": total,
        "accessibility_result_count": len(rows),
        "accessibility_ratio": ratio,
        "accessibility_results": rows,
        "recommendation": recommendation,
    }
