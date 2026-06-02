from __future__ import annotations

from graph.store.source_alternate_link_summary import summarize_source_alternate_links


def test_source_alternate_link_summary_reads_metadata_lists_and_html_tags():
    summary = summarize_source_alternate_links(
        [
            {
                "source_id": "a",
                "metadata": {
                    "alternate_links": [
                        {"href": "https://example.com/en", "hreflang": "EN-US", "type": "Text/HTML"},
                        {"url": "https://example.com/feed", "type": "Application/RSS+XML"},
                    ]
                },
            },
            {
                "source_id": "b",
                "content": '<html><link rel="canonical alternate" hreflang="ja_JP" type="text/html" href="/ja"></html>',
            },
            {"source_id": "c", "metadata": {"hreflang_links": {"fr": "https://example.com/fr"}}},
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_alternate_links"] == 3
    assert summary["total_alternate_links"] == 4
    assert summary["hreflang_counts"] == {"en-us": 1, "fr": 1, "ja-jp": 1}
    assert summary["media_type_counts"] == {"application/rss+xml": 1, "text/html": 2}
    assert summary["missing_alternate_link_count"] == 1
    assert summary["samples"] == [
        {
            "source_id": "a",
            "alternates": [
                {"href": "https://example.com/en", "hreflang": "en-us", "type": "text/html"},
                {"href": "https://example.com/feed", "hreflang": "", "type": "application/rss+xml"},
            ],
        },
        {"source_id": "b", "alternates": [{"href": "/ja", "hreflang": "ja-jp", "type": "text/html"}]},
    ]


def test_source_alternate_link_summary_bounds_samples_and_alternates():
    summary = summarize_source_alternate_links(
        [
            {"source_id": "b", "alternates": [{"href": "/b1"}, {"href": "/b2"}]},
            {"source_id": "a", "alternates": [{"href": "/a1"}]},
        ],
        sample_limit=1,
    )

    assert summary["samples"] == [{"source_id": "a", "alternates": [{"href": "/a1", "hreflang": "", "type": ""}]}]
