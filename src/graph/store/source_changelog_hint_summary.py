"""Summarize changelog and release-note hints in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, source_id

_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("changelog", (r"\bchangelog\b", r"/changelog\b")),
    ("release_note", (r"\brelease\s+notes?\b", r"/releases?\b")),
    ("version_history", (r"\bversion\s+history\b",)),
    ("migration", (r"\bmigration\s+guides?\b",)),
    ("breaking_change", (r"\bbreaking\s+changes?\b",)),
    ("deprecation", (r"\bdeprecation\s+notices?\b", r"\bdeprecated\b")),
)
_URL_RE = re.compile(r"https?://[^\s)>\"]+", re.I)


def summarize_source_changelog_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    hinted = [row for row in rows if row["cue_categories"]]
    counts = Counter(category for row in hinted for category in row["cue_categories"])
    return {
        "total_sources": len(source_list),
        "sources_with_changelog_hints": len(hinted),
        "cue_counts": dict(sorted(counts.items())),
        "changelog_url_samples": [url for row in hinted for url in row["changelog_urls"]][: max(0, sample_limit)],
        "samples": sorted(hinted, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    text = _source_text(source)
    categories = [name for name, patterns in _CUES if any(re.search(pattern, text, re.I) for pattern in patterns)]
    urls = [url.rstrip(".,") for url in _URL_RE.findall(text) if re.search(r"changelog|release|version|migration", url, re.I)]
    return {"source_id": source_id(source) or str(index), "url": field_value(get(source, "url") or get(source, "source_url")), "cue_categories": categories, "changelog_urls": urls}


def _source_text(source: Mapping[str, Any] | object) -> str:
    values = [get(source, key) for key in ("url", "source_url", "title", "content", "text", "snippet", "description")]
    values.extend(flatten_values(metadata(source)))
    return " ".join(field_value(value) for value in values if field_value(value))
