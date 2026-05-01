"""Lightweight local keyword extraction for graph units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.types.models import KnowledgeUnit

TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_WEIGHT = 3
CONTENT_WEIGHT = 1
TAG_WEIGHT = 2

COMMON_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "don",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in stopwords]


def _normalized_stopwords(stopwords: Iterable[str] | None) -> set[str]:
    normalized = set(COMMON_STOPWORDS)
    for value in stopwords or ():
        normalized.update(TOKEN_RE.findall(str(value).lower()))
    return normalized


def extract_keywords(
    units: Iterable[KnowledgeUnit],
    *,
    limit: int = 20,
    min_count: int = 1,
    include_tags: bool = True,
    stopwords: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return ranked local keywords extracted from unit titles, content, and tags."""
    if limit <= 0:
        return []

    ignored = _normalized_stopwords(stopwords)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"title": 0, "content": 0, "tags": 0})
    unit_ids: dict[str, set[str]] = defaultdict(set)

    for unit in units:
        unit_id = str(unit.id)
        fields = (
            ("title", unit.title),
            ("content", unit.content),
        )
        for source, text in fields:
            for token in _tokenize(text or "", ignored):
                counts[token][source] += 1
                if unit_id:
                    unit_ids[token].add(unit_id)

        if include_tags:
            for tag in unit.tags or []:
                for token in _tokenize(tag, ignored):
                    counts[token]["tags"] += 1
                    if unit_id:
                        unit_ids[token].add(unit_id)

    ranked = []
    for keyword, source_counts in counts.items():
        count = sum(source_counts.values())
        if count < min_count:
            continue
        score = (
            source_counts["title"] * TITLE_WEIGHT
            + source_counts["content"] * CONTENT_WEIGHT
            + source_counts["tags"] * TAG_WEIGHT
        )
        ranked.append(
            {
                "keyword": keyword,
                "score": score,
                "count": count,
                "sources": dict(source_counts),
                "unit_ids": sorted(unit_ids[keyword]),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], -item["count"], item["keyword"]))
    return ranked[:limit]
