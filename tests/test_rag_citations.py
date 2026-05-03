from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from graph.rag import format_result_citations


@dataclass
class ResultObject:
    title: str | None = None
    source: str | None = None
    url: str | None = None
    author: str | list[str] | None = None
    date: str | None = None
    metadata: dict = field(default_factory=dict)


def test_format_result_citations_builds_inline_citations_from_dicts_and_objects():
    citations = format_result_citations(
        [
            {
                "title": "Storage incentives",
                "source": "Energy Review",
                "url": "https://example.com/storage",
                "author": "A. Smith",
                "date": "2026-01-15T10:30:00+00:00",
            },
            ResultObject(
                title="Grid resilience",
                source="Grid Journal",
                url="https://grid.example/report",
                author=["R. Chen", "M. Patel"],
                date="2025-12-20",
            ),
        ]
    )

    assert citations == [
        "[Storage incentives; Energy Review, A. Smith, 2026-01-15, https://example.com/storage]",
        "[Grid resilience; Grid Journal, R. Chen, M. Patel, 2025-12-20, https://grid.example/report]",
    ]


def test_format_result_citations_builds_markdown_footnotes():
    citations = format_result_citations(
        [
            {
                "title": "Battery adoption",
                "metadata": {
                    "source_name": "Market Notes",
                    "source_url": "https://example.com/battery",
                    "publication_date": datetime(2026, 2, 5, 9, 0),
                },
            },
            {
                "title": "Transmission queue",
                "source": "Utility Data",
                "url": "https://example.com/queue",
            },
        ],
        style="footnote",
    )

    assert citations == [
        "[^1]: Battery adoption. Market Notes 2026-02-05 https://example.com/battery",
        "[^2]: Transmission queue. Utility Data https://example.com/queue",
    ]


def test_format_result_citations_tolerates_missing_fields():
    citations = format_result_citations(
        [
            {"id": "local-note"},
            {"source": "Archive Only"},
            {"url": "https://example.com/no-title"},
        ]
    )

    assert citations == [
        "[local-note]",
        "[Archive Only]",
        "[https://example.com/no-title]",
    ]


def test_format_result_citations_collapses_duplicate_urls_deterministically():
    citations = format_result_citations(
        [
            {
                "title": "Original title",
                "source": "Example",
                "url": "https://EXAMPLE.com/report/#section",
            },
            {
                "title": "Duplicate with richer metadata",
                "source": "Example",
                "author": "Later Author",
                "url": "https://example.com/report",
            },
            {
                "title": "Different query",
                "source": "Example",
                "url": "https://example.com/report?version=2",
            },
        ]
    )

    assert citations == [
        "[Original title; Example, https://EXAMPLE.com/report/#section]",
        "[Different query; Example, https://example.com/report?version=2]",
    ]


def test_format_result_citations_preserves_stable_first_seen_order():
    results = [
        {"title": "Beta", "url": "https://example.com/b"},
        {"title": "Alpha", "url": "https://example.com/a"},
        {"title": "Beta duplicate", "url": "https://example.com/b/"},
        {"title": "Gamma", "url": "https://example.com/g"},
    ]

    assert format_result_citations(results) == [
        "[Beta; https://example.com/b]",
        "[Alpha; https://example.com/a]",
        "[Gamma; https://example.com/g]",
    ]


def test_format_result_citations_supports_nested_unit_metadata_and_tuple_results():
    unit = ResultObject(
        title="Nested title",
        metadata={
            "source_name": "Nested Source",
            "source_url": "https://example.com/nested",
        },
    )

    citations = format_result_citations([({"unit": unit}, 0.97)])

    assert citations == ["[Nested title; Nested Source, https://example.com/nested]"]


def test_format_result_citations_rejects_unknown_styles():
    with pytest.raises(ValueError, match="style must be one of"):
        format_result_citations([], style="apa")
