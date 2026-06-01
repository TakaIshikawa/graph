"""Analyze how retrieved evidence records were accessed."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_id, string, value
from graph.rag._record_text import text_blob

ACCESS_METHODS = ("api", "rss", "csv", "pdf", "html", "web", "manual", "export", "archive", "database")
_METHOD_FIELDS = ("method", "access_method", "retrieval_method", "source_type")
_ALIASES: dict[str, tuple[str, ...]] = {
    "api": ("api", "rest", "graphql"),
    "rss": ("rss", "atom"),
    "csv": ("csv", "comma separated values", "comma-separated values"),
    "pdf": ("pdf", "portable document format"),
    "html": ("html", "html page", "markup"),
    "web": ("web", "website", "webpage", "web page", "browser", "url", "http", "https"),
    "manual": ("manual", "manually", "human entered", "human-entered", "hand collected"),
    "export": ("export", "exported", "data export"),
    "archive": ("archive", "archived", "snapshot", "wayback"),
    "database": ("database", "db", "sql", "warehouse"),
}
_ALIAS_TO_METHOD = {alias: method for method, aliases in _ALIASES.items() for alias in aliases}
_CUE_RE = re.compile(
    r"\b(?:api|rest|graphql|rss|atom|csv|comma[-\s]+separated\s+values|pdf|portable\s+document\s+format|"
    r"html|markup|web(?:site|page)?|web\s+page|browser|url|https?|manual(?:ly)?|human[-\s]+entered|"
    r"hand\s+collected|export(?:ed)?|data\s+export|archive(?:d)?|snapshot|wayback|database|db|sql|warehouse)\b",
    re.I,
)


def analyze_evidence_access_methods(evidence: Iterable[Any] | None = None, sample_limit: int = 5) -> dict[str, Any]:
    """Return aggregate access-method counts and capped record samples."""
    records = list(evidence or [])
    limit = max(0, int(sample_limit))
    method_counts: Counter[str] = Counter({method: 0 for method in ACCESS_METHODS})
    samples: list[dict[str, str]] = []
    unknown_count = 0

    for index, record in enumerate(records):
        method = _metadata_method(record) or _text_method(record)
        if method is None:
            method = "unknown"
            unknown_count += 1
        else:
            method_counts[method] += 1

        if len(samples) < limit:
            samples.append({"result_id": result_id(record, index), "method": method})

    return {
        "record_count": len(records),
        "method_counts": dict(method_counts),
        "method_diversity": sum(1 for count in method_counts.values() if count > 0),
        "unknown_method_count": unknown_count,
        "samples": samples,
    }


def _metadata_method(record: Any) -> str | None:
    for field in _METHOD_FIELDS:
        method = _method_for_text(string(value(record, field)))
        if method is not None:
            return method
    return None


def _text_method(record: Any) -> str | None:
    return _method_for_text(text_blob(record))


def _method_for_text(text: str | None) -> str | None:
    if not text:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    if normalized in _ALIAS_TO_METHOD:
        return _ALIAS_TO_METHOD[normalized]

    match = _CUE_RE.search(text)
    if match is None:
        return None
    cue = re.sub(r"[^a-z0-9]+", " ", match.group(0).casefold()).strip()
    return _ALIAS_TO_METHOD.get(cue)
