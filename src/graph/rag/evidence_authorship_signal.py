"""Summarize authorship signals across RAG evidence."""

from __future__ import annotations

from typing import Any

from graph.rag._analysis_utils import result_id, string, value

_AUTHOR_KEYS = ("author", "authors", "creator", "byline", "organization")


def analyze_evidence_authorship_signals(evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Return counts for evidence items with author-like metadata."""
    total = len(evidence_items or [])
    authored_count = 0
    missing_items = []
    by_author: dict[str, int] = {}
    for index, item in enumerate(evidence_items or []):
        authors = _authors(item)
        if authors:
            authored_count += 1
            for author in authors:
                by_author[author] = by_author.get(author, 0) + 1
        else:
            missing_items.append({"item_id": result_id(item, index), "index": index})
    missing_count = total - authored_count
    return {
        "total_items": total,
        "authored_count": authored_count,
        "missing_authorship_count": missing_count,
        "authored_ratio": 0.0 if total == 0 else round(authored_count / total, 4),
        "missing_items": missing_items,
        "by_author": dict(sorted(by_author.items())),
    }


def _authors(item: Any) -> list[str]:
    seen: set[str] = set()
    authors = []
    for key in _AUTHOR_KEYS:
        raw = value(item, key)
        values = raw if isinstance(raw, list | tuple | set) else [raw]
        for candidate in values:
            text = string(candidate)
            if text and text not in seen:
                seen.add(text)
                authors.append(text)
    return authors
