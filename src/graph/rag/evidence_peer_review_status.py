"""Classify evidence items by scholarly peer-review status cues."""

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

STATUSES = ("peer_reviewed", "preprint", "report", "news_or_blog", "documentation", "unknown")
_STATUS_ORDER = {status: index for index, status in enumerate(STATUSES)}
_TEXT_KEYS = ("title", "source", "source_name", "publisher", "venue", "journal", "publication", "url", "domain")
_STATUS_KEYS = (
    "peer_review_status",
    "review_status",
    "publication_status",
    "publication_type",
    "source_type",
    "document_type",
    "type",
)
_EXPLICIT_VALUES: dict[str, str] = {
    "peer_reviewed": "peer_reviewed",
    "peer reviewed": "peer_reviewed",
    "refereed": "peer_reviewed",
    "journal article": "peer_reviewed",
    "journal_article": "peer_reviewed",
    "preprint": "preprint",
    "working paper": "preprint",
    "working_paper": "preprint",
    "report": "report",
    "technical report": "report",
    "technical_report": "report",
    "white paper": "report",
    "white_paper": "report",
    "news": "news_or_blog",
    "blog": "news_or_blog",
    "blog post": "news_or_blog",
    "blog_post": "news_or_blog",
    "documentation": "documentation",
    "docs": "documentation",
    "manual": "documentation",
    "api reference": "documentation",
    "api_reference": "documentation",
}
_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "preprint": re.compile(
        r"\b(?:preprint|arxiv|bioRxiv|medRxiv|chemRxiv|ssrn|research square|osf preprints?|"
        r"working paper)\b",
        re.I,
    ),
    "peer_reviewed": re.compile(
        r"\b(?:peer[- ]reviewed|refereed|journal article|published in (?:the )?journal|"
        r"journal of|proceedings of|transactions on|pubmed|pmid|doi:)\b",
        re.I,
    ),
    "documentation": re.compile(
        r"\b(?:documentation|docs|api reference|developer guide|user guide|manual|reference manual|"
        r"readthedocs|release notes|changelog)\b",
        re.I,
    ),
    "report": re.compile(
        r"\b(?:technical report|research report|policy report|white paper|government report|"
        r"annual report|staff report|case report|report no\.?|working paper)\b",
        re.I,
    ),
    "news_or_blog": re.compile(
        r"\b(?:news|blog|blog post|newsletter|op-ed|opinion|editorial|press release|"
        r"reported by|reuters|ap news|medium|substack)\b",
        re.I,
    ),
}
_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "preprint": (
        "arxiv.org",
        "biorxiv.org",
        "medrxiv.org",
        "chemrxiv.org",
        "ssrn.com",
        "researchsquare.com",
        "osf.io",
    ),
    "peer_reviewed": (
        "pubmed.ncbi.nlm.nih.gov",
        "jstor.org",
        "sciencedirect.com",
        "springer.com",
        "wiley.com",
        "tandfonline.com",
        "ieeexplore.ieee.org",
        "acm.org",
        "nature.com",
        "science.org",
    ),
    "documentation": ("docs.", "developer.", "developers.", "readthedocs.io", "devdocs.io"),
    "news_or_blog": ("reuters.com", "apnews.com", "nytimes.com", "bbc.", "theguardian.com", "medium.com", "substack.com", "blog."),
}


def classify_evidence_peer_review_status(evidence_items: Iterable[Any] | None) -> dict[str, Any]:
    """Return peer-review status counts and per-evidence classifications."""
    try:
        items = list(evidence_items or [])
    except TypeError:
        items = []

    rows = []
    counts: Counter[str] = Counter()
    for index, item in enumerate(items):
        status, reasons = _classify(item)
        counts[status] += 1
        rows.append(
            {
                "evidence_id": result_id(item, index),
                "peer_review_status": status,
                "reasons": reasons,
            }
        )

    status_counts = {status: counts.get(status, 0) for status in STATUSES}
    total = len(rows)
    return {
        "total_evidence": total,
        "status_counts": status_counts,
        "status_share": {status: rounded_ratio(count, total) for status, count in status_counts.items()},
        "per_evidence": rows,
    }


def _classify(item: Any) -> tuple[str, list[str]]:
    explicit = _explicit_status(item)
    if explicit is not None:
        return explicit

    cues = _cue_matches(item)
    if not cues:
        if value(item, "peer_reviewed") is False:
            return "unknown", ["metadata_peer_reviewed_false"]
        return "unknown", ["insufficient_peer_review_signals"]

    status = min(cues, key=lambda name: (-len(cues[name]), _STATUS_ORDER[name]))
    return status, cues[status]


def _explicit_status(item: Any) -> tuple[str, list[str]] | None:
    for key in _STATUS_KEYS:
        text = string(value(item, key))
        if not text:
            continue
        normalized = "_".join(text.casefold().split())
        spaced = " ".join(text.casefold().split())
        status = _EXPLICIT_VALUES.get(normalized) or _EXPLICIT_VALUES.get(spaced)
        if status is not None:
            return status, [f"metadata_{key}_{status}"]

    reviewed = value(item, "peer_reviewed")
    if reviewed is True:
        return "peer_reviewed", ["metadata_peer_reviewed_true"]
    return None


def _cue_matches(item: Any) -> dict[str, list[str]]:
    fields = {
        "title_text": content_text(item),
        "source_venue": _source_venue_text(item),
        "metadata": _metadata_text(item),
    }
    cues: dict[str, list[str]] = {status: [] for status in STATUSES if status != "unknown"}

    for status, pattern in _TEXT_PATTERNS.items():
        for field, text in fields.items():
            if text and pattern.search(text):
                cues[status].append(field)

    domain = domain_for(item) or ""
    source = fields["source_venue"]
    for status, hints in _DOMAIN_HINTS.items():
        if any(hint in domain or hint in source for hint in hints):
            cues[status].append("domain")

    return {status: sorted(set(reasons)) for status, reasons in cues.items() if reasons}


def _source_venue_text(item: Any) -> str:
    parts = [domain_for(item) or ""]
    parts.extend(string(value(item, key)) or "" for key in _TEXT_KEYS)
    return " ".join(parts).casefold()


def _metadata_text(item: Any) -> str:
    return " ".join(text.casefold() for text in iter_strings(metadata(item)))
