"""Summarize source status-page hints."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, source_id

_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("status_page", (r"\bstatuspage\.io\b", r"/status\b", r"\bstatus\s+page\b")),
    ("incident_history", (r"\bincident\s+history\b", r"\bpast\s+incidents?\b")),
    ("uptime", (r"\buptime\b",)),
    ("degraded_performance", (r"\bdegraded\s+performance\b",)),
    ("maintenance", (r"\bmaintenance\s+(?:notice|window|scheduled|event)s?\b", r"\bscheduled\s+maintenance\b")),
    ("incident", (r"\bincidents?\b",)),
)


def summarize_source_status_pages(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    hinted = [row for row in rows if row["cue_categories"]]
    counts = Counter(category for row in hinted for category in row["cue_categories"])
    return {
        "total_sources": len(source_list),
        "sources_with_status_page_hints": len(hinted),
        "status_page_link_count": counts.get("status_page", 0),
        "incident_history_hint_count": counts.get("incident_history", 0),
        "cue_counts": dict(sorted(counts.items())),
        "samples": sorted(hinted, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    text = _source_text(source)
    categories = [name for name, patterns in _CUES if any(re.search(pattern, text, re.I) for pattern in patterns)]
    return {"source_id": source_id(source) or str(index), "title": field_value(get(source, "title")), "url": field_value(get(source, "url") or get(source, "source_url")), "cue_categories": categories}


def _source_text(source: Mapping[str, Any] | object) -> str:
    values = [get(source, key) for key in ("url", "source_url", "title", "content", "text", "snippet", "description")]
    values.extend(flatten_values(metadata(source)))
    return " ".join(field_value(value) for value in values if field_value(value))
