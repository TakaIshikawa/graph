"""Summarize privacy, terms, and policy links in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, source_id

_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("privacy", (r"\bprivacy\s+policy\b", r"/privacy\b")),
    ("terms", (r"\bterms\s+of\s+service\b", r"\bterms\s+and\s+conditions\b", r"/terms\b")),
    ("dpa", (r"\bdata\s+processing\s+(?:agreement|addendum)\b", r"\bdpa\b")),
    ("subprocessor", (r"\bsubprocessors?\b",)),
    ("cookie", (r"\bcookie\s+policy\b", r"/cookies?\b")),
    ("acceptable_use", (r"\bacceptable\s+use\s+policy\b", r"\baup\b")),
)
_URL_RE = re.compile(r"https?://[^\s)>\"]+", re.I)


def summarize_source_privacy_terms_links(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    hinted = [row for row in rows if row["cue_categories"]]
    counts = Counter(category for row in hinted for category in row["cue_categories"])
    return {
        "total_sources": len(source_list),
        "sources_with_policy_hints": len(hinted),
        "cue_counts": dict(sorted(counts.items())),
        "policy_url_samples": [url for row in hinted for url in row["policy_urls"]][: max(0, sample_limit)],
        "samples": sorted(hinted, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    text = _source_text(source)
    categories = [name for name, patterns in _CUES if any(re.search(pattern, text, re.I) for pattern in patterns)]
    urls = [url.rstrip(".,") for url in _URL_RE.findall(text) if any(re.search(pattern, url, re.I) for _, patterns in _CUES for pattern in patterns)]
    return {"source_id": source_id(source) or str(index), "url": field_value(get(source, "url") or get(source, "source_url")), "cue_categories": categories, "policy_urls": urls}


def _source_text(source: Mapping[str, Any] | object) -> str:
    values = [get(source, key) for key in ("url", "source_url", "title", "content", "text", "snippet", "description")]
    values.extend(flatten_values(metadata(source)))
    return " ".join(field_value(value) for value in values if field_value(value))
