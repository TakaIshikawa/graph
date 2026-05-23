"""Summarize evidence method mix across retrieved RAG results."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import (
    content_text,
    domain_for,
    iter_strings,
    metadata,
    result_id,
    rounded_ratio,
    string,
    value,
)

METHODS = (
    "empirical_study",
    "official_documentation",
    "news_reporting",
    "opinion",
    "forum",
    "reference",
    "unknown",
)

_METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "empirical_study": re.compile(
        r"\b(?:study|experiment|trial|survey|participants?|respondents?|dataset|data set|"
        r"sample size|n\s*=\s*\d+|p-value|confidence interval|methodology|peer[- ]reviewed)\b",
        re.I,
    ),
    "official_documentation": re.compile(
        r"\b(?:official|documentation|docs|developer guide|api reference|manual|release notes|"
        r"specification|policy|standard|rfc|changelog)\b",
        re.I,
    ),
    "news_reporting": re.compile(
        r"\b(?:news|reported|reporting|reporter|press release|wire service|according to|"
        r"investigation|interviewed|announced)\b",
        re.I,
    ),
    "opinion": re.compile(
        r"\b(?:opinion|editorial|op-ed|commentary|analysis|essay|review|i think|we think|"
        r"perspective|takeaway|my view)\b",
        re.I,
    ),
    "forum": re.compile(
        r"\b(?:forum|thread|comment|comments|discussion|q&a|question|answer|stackoverflow|"
        r"reddit|community)\b",
        re.I,
    ),
    "reference": re.compile(
        r"\b(?:encyclopedia|dictionary|glossary|definition|reference|handbook|wiki|catalog|"
        r"database|registry|index)\b",
        re.I,
    ),
}
_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "official_documentation": (
        "docs.",
        "developer.",
        "developers.",
        "dev.",
        ".gov",
        ".edu",
        "ietf.org",
        "w3.org",
        "iso.org",
    ),
    "news_reporting": (
        "apnews.com",
        "reuters.com",
        "nytimes.com",
        "washingtonpost.com",
        "bbc.",
        "cnn.com",
        "theguardian.com",
        "wsj.com",
        "bloomberg.com",
    ),
    "forum": (
        "reddit.com",
        "news.ycombinator.com",
        "stackoverflow.com",
        "stackexchange.com",
        "discourse.",
        "community.",
        "forum.",
    ),
    "reference": (
        "wikipedia.org",
        "wikidata.org",
        "britannica.com",
        "dictionary.com",
        "reference.com",
    ),
}
_TAGS_KEYS = ("tags", "tag", "labels", "categories")
_SOURCE_KEYS = (
    "source",
    "source_project",
    "source_name",
    "source_type",
    "type",
    "kind",
    "publisher",
    "author",
    "url",
)


def _normalized_text(value_: Any) -> str:
    return " ".join(text.casefold() for text in iter_strings(value_))


def _tag_text(result: Any) -> str:
    return " ".join(_normalized_text(value(result, key)) for key in _TAGS_KEYS)


def _source_text(result: Any) -> str:
    values = [domain_for(result) or ""]
    values.extend(string(value(result, key)) or "" for key in _SOURCE_KEYS)
    return " ".join(values).casefold()


def _metadata_text(result: Any) -> str:
    meta = metadata(result)
    return _normalized_text(meta)


def _cue_matches(result: Any) -> dict[str, list[str]]:
    fields = {
        "title_content": content_text(result),
        "source": _source_text(result),
        "tags": _tag_text(result),
        "metadata": _metadata_text(result),
    }
    cues: dict[str, list[str]] = {method: [] for method in METHODS if method != "unknown"}

    for method, pattern in _TEXT_PATTERNS.items():
        for field, text in fields.items():
            if text and pattern.search(text):
                cues[method].append(field)

    source = fields["source"]
    for method, hints in _DOMAIN_HINTS.items():
        if any(hint in source for hint in hints):
            cues[method].append("domain")

    return {method: sorted(set(matches)) for method, matches in cues.items() if matches}


def _classify(result: Any) -> tuple[str, list[str]]:
    cues = _cue_matches(result)
    if not cues:
        return "unknown", []

    method = min(cues, key=lambda name: (-len(cues[name]), _METHOD_ORDER[name]))
    return method, cues[method]


def _diversity_score(counter: Counter[str], total: int) -> float:
    if total <= 1 or len(counter) <= 1:
        return 0.0
    raw = 1.0 - sum((count / total) ** 2 for count in counter.values())
    max_methods = min(total, len(METHODS))
    max_raw = 1.0 - (1.0 / max_methods)
    return round(raw / max_raw, 4) if max_raw else 0.0


def _dominant(counter: Counter[str]) -> str:
    if not counter:
        return "unknown"
    return min(counter, key=lambda method: (-counter[method], _METHOD_ORDER[method]))


def analyze_result_evidence_method_mix(results: Iterable[Any] | None) -> dict[str, Any]:
    """Return method counts, dominance, diversity, and per-result classifications."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    method_counts: Counter[str] = Counter()
    classifications: list[dict[str, Any]] = []

    for index, result in enumerate(rows):
        method, cues = _classify(result)
        method_counts[method] += 1
        classifications.append(
            {
                "result_id": result_id(result, index),
                "method": method,
                "cues": cues,
            }
        )

    counts = {method: method_counts.get(method, 0) for method in METHODS}
    populated_counts = Counter({method: count for method, count in counts.items() if count})
    total = len(rows)

    return {
        "total_results": total,
        "method_counts": counts,
        "method_share": {method: rounded_ratio(count, total) for method, count in counts.items()},
        "dominant_method": _dominant(populated_counts),
        "diversity_score": _diversity_score(populated_counts, total),
        "per_result": classifications,
    }
